from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import torch
from torch import Tensor, nn

from cusrl.module import Actor, AdaptiveNormalDist, NormalDist
from cusrl.module.distribution import MeanStdDict
from cusrl.template import ActorCritic, Hook
from cusrl.utils.dict_utils import prefix_dict_keys
from cusrl.utils.nest import map_nested
from cusrl.utils.typing import Memory, NestedTensor, Slice

__all__ = [
    # Elements
    "SymmetricActor",
    "SymmetryDef",
    "SymmetryDefLike",
    # Hooks
    "SymmetryHook",
    "SymmetryLoss",
    "SymmetricArchitecture",
    "SymmetricDataAugmentation",
]


class SymmetryDef:
    def __init__(
        self,
        destination_indices: Sequence[int],
        flipped_indices: Sequence[int],
    ):
        self.destination_indices = destination_indices
        self.flipped_indices = flipped_indices

        self.destination = torch.tensor(destination_indices, dtype=torch.long)
        self.multiplier = torch.ones(len(destination_indices))
        self.multiplier[flipped_indices] = -1.0

    def __call__(self, input: Tensor):
        self.destination = self.destination.to(input.device)
        self.multiplier = self.multiplier.to(dtype=input.dtype, device=input.device)
        return input[..., self.destination] * self.multiplier

    def __repr__(self):
        return f"SymmetryDef(destination_indices={self.destination_indices}, flipped_indices={self.flipped_indices})"


SymmetryDefLike: TypeAlias = Callable[[Tensor], Tensor]


class SymmetryHook(Hook[ActorCritic]):
    def __init__(self):
        super().__init__()
        self.mirror_observation: SymmetryDefLike
        self.mirror_state: SymmetryDefLike | None
        self.mirror_action: SymmetryDefLike

    def init(self):
        num_symmetry_hooks = sum(isinstance(hook, SymmetryHook) for hook in self.agent.hook)
        if num_symmetry_hooks > 1:
            raise ValueError("At most one symmetry hook should be registered.")

        if self.agent.environment_spec.mirror_observation is None:
            raise ValueError("'mirror_observation' should be defined for symmetry hooks.")
        self.mirror_observation = self.agent.environment_spec.mirror_observation

        if self.agent.has_state and self.agent.environment_spec.mirror_state is None:
            raise ValueError("'mirror_state' should be defined for symmetry hooks.")
        self.mirror_state = self.agent.environment_spec.mirror_state

        if self.agent.environment_spec.mirror_action is None:
            raise ValueError("'mirror_action' should be defined for symmetry hooks.")
        self.mirror_action = self.agent.environment_spec.mirror_action


class SymmetryLoss(SymmetryHook):
    """Implements a symmetry loss to facilitate symmetry in the action
    distribution.

    Described in "Learning Symmetric and Low-Energy Locomotion",
    https://dl.acm.org/doi/abs/10.1145/3197517.3201397

    Args:
        weight (float | None):
            Scaling factor for the symmetry loss. If None, the symmetry loss is
            not applied.
        symmetrize_action_std (bool, optional):
            Whether to symmetrize the action standard deviation. Defaults to
            False.
    """

    def __init__(self, weight: float | None, symmetrize_action_std: bool = False):
        if weight is not None and weight < 0:
            raise ValueError("'weight' must be None or non-negative.")
        super().__init__()
        self.symmetrize_action_std = symmetrize_action_std

        # Mutable attributes
        self.weight: float | None
        self.register_mutable("weight", weight)

        # Runtime attributes
        self.criterion: nn.MSELoss
        self.mirrored_actor_memory: Memory

    def init(self):
        super().init()
        self.criterion = nn.MSELoss()
        self.mirrored_actor_memory = None

    @torch.no_grad()
    def post_step(self, transition):
        actor = self.agent.actor
        observation = cast(Tensor, transition["observation"])
        done = cast(Tensor, transition["done"])

        mirrored_observation = self.mirror_observation(observation)
        transition["mirrored_actor_memory"] = self.mirrored_actor_memory
        with self.agent.autocast():
            self.mirrored_actor_memory = actor.step_memory(
                mirrored_observation,
                memory=self.mirrored_actor_memory,
            )
            actor.reset_memory(self.mirrored_actor_memory, done)

    def objective(self, batch):
        if self.weight is None:
            return None

        actor = self.agent.actor
        observation = cast(Tensor, batch["observation"])
        with self.agent.autocast():
            mirrored_action_dist, _ = actor(
                self.mirror_observation(observation),
                memory=batch.get("mirrored_actor_memory"),
                done=batch["done"],
            )

        curr_action_dist = cast(MeanStdDict, batch["curr_action_dist"])
        loss = self.criterion(
            curr_action_dist["mean"],
            self.mirror_action(mirrored_action_dist["mean"]),
        )
        if self.symmetrize_action_std:
            loss += self.criterion(
                curr_action_dist["std"],
                self.mirror_action(mirrored_action_dist["std"]).abs(),
            )
        symmetry_loss = self.weight * loss
        self.agent.record(symmetry_loss=symmetry_loss)
        return symmetry_loss


class SymmetricDataAugmentation(SymmetryHook):
    """Augments training data by adding mirrored transitions to the batch.

    Described in "Symmetry Considerations for Learning Task Symmetric Robot
    Policies",
    https://ieeexplore.ieee.org/abstract/document/10611493

    This hook doubles the effective batch size by appending a mirrored version
    of each transition. For each transition (s, a, r, s'), it adds a
    corresponding mirrored transition (s_m, a_m, r, s'_m), where _m denotes a
    mirrored version. This encourages the learned policy to be symmetric.

    It also manages the recurrent state (memory) for the actor when processing
    mirrored observations, ensuring correct backpropagation through time for
    recurrent policies.

    Args:
        augments_value (bool):
            Whether to augment the value function with mirrored transitions.
            Defaults to True.
    """

    def __init__(self, augments_value: bool = True):
        self.augments_value = augments_value
        super().__init__()

        # Runtime attributes
        self.mirrored_actor_memory: Memory
        self.mirrored_critic_memory: Memory

    def init(self):
        super().init()
        self.mirrored_actor_memory = None
        self.mirrored_critic_memory = None

    @torch.no_grad()
    def post_step(self, transition):
        # Augment observation and next_observation
        observation = cast(Tensor, transition["observation"])
        mirrored_observation, transition["augmented_observation"] = self._build_augmented_tensor(
            observation, self.mirror_observation
        )
        _, transition["augmented_next_observation"] = self._build_augmented_tensor(
            cast(Tensor, transition["next_observation"]), self.mirror_observation
        )

        # Augment state and next_state if available
        if (state := cast(torch.Tensor | None, transition.get("state"))) is not None:
            assert self.mirror_state is not None
            mirrored_state, transition["augmented_state"] = self._build_augmented_tensor(state, self.mirror_state)
            _, transition["augmented_next_state"] = self._build_augmented_tensor(
                cast(torch.Tensor, transition["next_state"]), self.mirror_state
            )
        else:
            mirrored_state = mirrored_observation

        # Augment action
        _, transition["augmented_action"] = self._build_augmented_tensor(
            cast(Tensor, transition["action"]), self.mirror_action
        )

        # Augment memory for actor
        actor, critic = self.agent.actor, self.agent.critic
        done = cast(Tensor, transition["done"])
        with self.agent.autocast():
            if self.mirrored_actor_memory is not None:
                transition["augmented_actor_memory"] = self._concat_memory(
                    cast(Memory, map_nested(lambda x: x.unsqueeze(1), transition["actor_memory"])),
                    self.mirrored_actor_memory,
                )
            self.mirrored_actor_memory = actor.step_memory(
                mirrored_observation.unsqueeze(0), self.mirrored_actor_memory
            )
            actor.reset_memory(self.mirrored_actor_memory, done)

        # Augment memory for critic if needed
        if self.augments_value:
            if self.mirrored_critic_memory is not None:
                transition["augmented_critic_memory"] = self._concat_memory(
                    cast(Memory, map_nested(lambda x: x.unsqueeze(1), transition["critic_memory"])),
                    self.mirrored_critic_memory,
                )
            self.mirrored_critic_memory = critic.step_memory(mirrored_state.unsqueeze(0), self.mirrored_critic_memory)
            critic.reset_memory(self.mirrored_critic_memory, done)

    def objective(self, batch):
        augmented_observation = cast(Tensor, batch["augmented_observation"])
        batch["observation"] = augmented_observation
        batch["next_observation"] = batch["augmented_next_observation"]
        batch["action"] = batch["augmented_action"]
        if self.agent.has_state:
            batch["state"] = batch["augmented_state"]
            batch["next_state"] = batch["augmented_next_state"]

        for key in ("action_logp", "advantage"):
            original = cast(Tensor, batch[key])
            batch[key] = original.unsqueeze(-3).expand(*augmented_observation.shape[:-1], original.size(-1))
        if (augmented_actor_memory := batch.get("augmented_actor_memory")) is not None:
            batch["actor_memory"] = augmented_actor_memory

        if self.augments_value:
            for key in ("value", "return"):
                original = cast(Tensor, batch[key])
                batch[key] = original.unsqueeze(-3).expand(*augmented_observation.shape[:-1], original.size(-1))
            if (augmented_critic_memory := batch.get("augmented_critic_memory")) is not None:
                batch["critic_memory"] = augmented_critic_memory

    @staticmethod
    def _build_augmented_tensor(original: Tensor, mirror: Callable[[Tensor], Tensor]) -> tuple[Tensor, Tensor]:
        mirrored = mirror(original).reshape(-1, *original.shape)
        return mirrored, torch.cat([original.unsqueeze(0), mirrored], dim=0)

    def _concat_memory(self, memory1: Memory, memory2: Memory):
        if memory1 is None:
            return None
        if isinstance(memory1, torch.Tensor):
            return torch.cat([memory1, memory2], dim=-3)
        return tuple(self._concat_memory(m1, m2) for m1, m2 in zip(memory1, memory2))


class SymmetricArchitecture(SymmetryHook):
    """Enforces a symmetric architecture on the agent's actor.

    Described in "On Learning Symmetric Locomotion",
    https://dl.acm.org/doi/abs/10.1145/3359566.3360070

    This hook wraps the agent's original actor with a `SymmetricActor` during
    the initialization phase, ensuring that the policy is strictly symmetric.
    """

    def pre_init(self, agent: ActorCritic):
        super().pre_init(agent)
        agent.actor_factory = SymmetricActorFactory(
            agent.actor_factory.backbone_factory,
            agent.actor_factory.distribution_factory,
            agent.actor_factory.latent_dim,
            mirror_observation=agent.environment_spec.mirror_observation,
            mirror_action=agent.environment_spec.mirror_action,
        )


@dataclass
class SymmetricActorFactory(Actor.Factory):
    mirror_observation: SymmetryDefLike | None = None
    mirror_action: SymmetryDefLike | None = None

    def __call__(self, input_dim: int | None, output_dim: int) -> Actor:
        actor = super().__call__(input_dim, output_dim)
        assert self.mirror_observation is not None, "mirror_observation must be defined."
        assert self.mirror_action is not None, "mirror_action must be defined."
        return SymmetricActor(
            actor,
            mirror_observation=self.mirror_observation,
            mirror_action=self.mirror_action,
        )


class SymmetricActor(Actor):
    def __init__(
        self,
        wrapped: Actor,
        mirror_observation: SymmetryDefLike,
        mirror_action: SymmetryDefLike,
    ):
        super().__init__(wrapped.backbone, wrapped.distribution)
        if not isinstance(self.distribution, (NormalDist, AdaptiveNormalDist)):
            raise ValueError("SymmetricActor can only be used with Normal distributions.")

        self.wrapped = wrapped
        self.mirror_observation = mirror_observation
        self.mirror_action = mirror_action
        self.is_distributed = self.wrapped.is_distributed

    def to_distributed(self):
        if not self.is_distributed:
            self.is_distributed = True
            self.wrapped = self.wrapped.to_distributed()
            self.backbone = self.wrapped.backbone
            self.distribution = self.wrapped.distribution
        return self

    def _forward_impl(
        self,
        observation: Tensor,
        memory: Memory = None,
        done: Tensor | None = None,
        backbone_kwargs: dict | None = None,
        distribution_kwargs: dict | None = None,
    ) -> tuple[NestedTensor, Memory]:
        if memory is not None:
            memory, mirrored_memory = memory
        else:
            memory = mirrored_memory = None

        self.wrapped.intermediate_repr.clear()
        mirrored_observation = self.mirror_observation(observation)
        mirrored_action_dist, mirrored_memory = self.wrapped(
            mirrored_observation,
            memory=mirrored_memory,
            done=done,
            backbone_kwargs=backbone_kwargs,
            distribution_kwargs=distribution_kwargs,
        )
        mirrored_intermediate_repr = self.wrapped.intermediate_repr

        self.wrapped.intermediate_repr = {}
        original_action_dist, memory = self.wrapped(
            observation,
            memory=memory,
            done=done,
            backbone_kwargs=backbone_kwargs,
            distribution_kwargs=distribution_kwargs,
        )

        self.intermediate_repr["original.action_dist"] = original_action_dist
        self.intermediate_repr.update(prefix_dict_keys(self.wrapped.intermediate_repr, "original."))
        self.intermediate_repr["mirrored.observation"] = mirrored_observation
        self.intermediate_repr["mirrored.action_dist"] = mirrored_action_dist
        self.intermediate_repr.update(prefix_dict_keys(mirrored_intermediate_repr, "mirrored."))
        action_dist = {
            "mean": (original_action_dist["mean"] + self.mirror_action(mirrored_action_dist["mean"])) / 2,
            "std": (original_action_dist["std"] + self.mirror_action(mirrored_action_dist["std"]).abs()) / 2,
        }
        if memory is None:
            return action_dist, None
        return action_dist, (memory, mirrored_memory)

    def _explore_impl(
        self,
        observation: Tensor,
        memory: Memory = None,
        deterministic: bool = False,
        backbone_kwargs: dict | None = None,
        distribution_kwargs: dict | None = None,
    ) -> tuple[NestedTensor, tuple[Tensor, Tensor], Memory]:
        action_dist, memory = self(
            observation,
            memory=memory,
            backbone_kwargs=backbone_kwargs,
            distribution_kwargs=distribution_kwargs,
        )
        if deterministic:
            original_action = self.distribution.determine(
                self.intermediate_repr["original.backbone.output"],
                observation=observation,
                **(distribution_kwargs or {}),
            )
            mirrored_action = self.distribution.determine(
                self.intermediate_repr["mirrored.backbone.output"],
                observation=self.intermediate_repr["mirrored.observation"],
                **(distribution_kwargs or {}),
            )
            action = (original_action + self.mirror_action(mirrored_action)) / 2
            logp = self.distribution.compute_logp(action_dist, action)
        else:
            action, logp = self.distribution.sample_from_dist(action_dist)
        return action_dist, (action, logp), memory

    def step_memory(self, observation, memory=None, **kwargs):
        if memory is not None:
            memory, mirrored_memory = memory
        else:
            memory = mirrored_memory = None

        memory = self.wrapped.step_memory(observation, memory=memory, **kwargs)
        mirrored_observation = self.mirror_observation(observation)
        mirrored_memory = self.wrapped.step_memory(mirrored_observation, memory=mirrored_memory, **kwargs)
        return None if memory is None else (memory, mirrored_memory)

    def reset_memory(self, memory: Memory, done: Slice | Tensor | None = None):
        if memory is None:
            return

        memory, mirrored_memory = memory
        self.wrapped.reset_memory(memory, done=done)
        self.wrapped.reset_memory(mirrored_memory, done=done)
