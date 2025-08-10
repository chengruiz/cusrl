from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from cusrl.module import Actor, AdaptiveNormalDist, NormalDist
from cusrl.module.distribution import MeanStdDict
from cusrl.template import ActorCritic, Hook
from cusrl.utils.dict_utils import prefix_dict_keys
from cusrl.utils.typing import Memory, NestedTensor, Slice

__all__ = [
    # Elements
    "SymmetricActor",
    "SymmetryDef",
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
        if self.destination.device != input.device:
            self.destination = self.destination.to(input.device)
            self.multiplier = self.multiplier.to(input.device)
        return input[..., self.destination] * self.multiplier

    def __repr__(self):
        return f"SymmetryDef(destination_indices={self.destination_indices}, flipped_indices={self.flipped_indices})"


class SymmetryHook(Hook[ActorCritic]):
    _mirror_observation: SymmetryDef
    _mirror_action: SymmetryDef

    def init(self):
        if self.agent.environment_spec.mirror_observation is None or self.agent.environment_spec.mirror_action is None:
            raise ValueError("'mirror_observation' and 'mirror_action' should be defined for symmetry hooks.")

        self._mirror_observation = self.agent.environment_spec.mirror_observation
        self._mirror_action = self.agent.environment_spec.mirror_action


class SymmetryLoss(SymmetryHook):
    """Implements a symmetry loss to facilitate symmetry in the action
    distribution.

    Described in "Learning Symmetric and Low-Energy Locomotion",
    https://dl.acm.org/doi/abs/10.1145/3197517.3201397

    Args:
        weight (float | None):
            Scaling factor for the symmetry loss. If None, the symmetry loss is
            not applied.
    """

    criterion: nn.MSELoss
    _mirrored_actor_memory: Memory

    # Mutable attributes
    weight: float | None

    def __init__(self, weight: float | None):
        super().__init__()
        self.register_mutable("weight", weight)

    def init(self):
        super().init()
        self.criterion = nn.MSELoss()
        self._mirrored_actor_memory = None

    @torch.no_grad()
    def post_step(self, transition):
        actor = self.agent.actor
        mirrored_observation = self._mirror_observation(transition["observation"])
        transition["mirrored_actor_memory"] = self._mirrored_actor_memory
        self._mirrored_actor_memory = actor.step_memory(mirrored_observation, memory=self._mirrored_actor_memory)
        actor.reset_memory(self._mirrored_actor_memory, transition["done"])

    def objective(self, batch):
        if self.weight is None:
            return None

        actor = self.agent.actor
        mirrored_action_dist, _ = actor(
            self._mirror_observation(batch["observation"]),
            memory=batch.get("mirrored_actor_memory"),
            done=batch["done"],
        )
        curr_action_dist = cast(MeanStdDict, batch["curr_action_dist"])

        mean_loss = self.criterion(curr_action_dist["mean"], self._mirror_action(mirrored_action_dist["mean"]))
        std_loss = self.criterion(curr_action_dist["std"], self._mirror_action(mirrored_action_dist["std"]))
        symmetry_loss = self.weight * (mean_loss + std_loss)
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
    """

    def __init__(self):
        super().__init__()
        self._mirrored_actor_memory = None

    @torch.no_grad()
    def post_step(self, transition):
        actor = self.agent.actor
        mirrored_observation = self._mirror_observation(transition["observation"])
        transition["mirrored_actor_memory"] = self._mirrored_actor_memory
        self._mirrored_actor_memory = actor.step_memory(mirrored_observation, memory=self._mirrored_actor_memory)
        actor.reset_memory(self._mirrored_actor_memory, transition["done"])

    def objective(self, batch):
        actor = self.agent.actor
        with self.agent.autocast():
            mirrored_action_dist, _ = actor(
                self._mirror_observation(batch["observation"]),
                memory=batch.get("mirrored_actor_memory"),
                done=batch["done"],
            )
            mirrored_action_logp = actor.compute_logp(mirrored_action_dist, self._mirror_action(batch["action"]))
            mirrored_entropy = actor.compute_entropy(mirrored_action_dist)
            mirrored_action_logp_diff = mirrored_action_logp - batch["action_logp"]

        batch["advantage"] = torch.cat([batch["advantage"], batch["advantage"]], dim=0)
        batch["action_logp_diff"] = torch.cat([batch["action_logp_diff"], mirrored_action_logp_diff], dim=0)
        batch["action_prob_ratio"] = torch.cat([batch["action_prob_ratio"], mirrored_action_logp_diff.exp()], dim=0)
        batch["curr_entropy"] = torch.cat([batch["curr_entropy"], mirrored_entropy], dim=0)


class SymmetricArchitecture(SymmetryHook):
    """Enforces a symmetric architecture on the agent's actor.

    Described in "On Learning Symmetric Locomotion",
    https://dl.acm.org/doi/abs/10.1145/3359566.3360070

    This hook wraps the agent's original actor with a `SymmetricActor` during
    the initialization phase, ensuring that the policy is strictly symmetric.
    """

    def init(self):
        super().init()
        self.agent.actor = SymmetricActor(self.agent.actor, self._mirror_observation, self._mirror_action)


class SymmetricActor(Actor):
    def __init__(
        self,
        wrapped: Actor,
        mirror_observation: SymmetryDef,
        mirror_action: SymmetryDef,
    ):
        super().__init__(wrapped.backbone, wrapped.distribution)
        if not isinstance(self.distribution, (NormalDist, AdaptiveNormalDist)):
            raise ValueError("SymmetricActor can only be used with Normal distributions.")

        self.wrapped = wrapped
        self._mirror_observation = mirror_observation
        self._mirror_action = mirror_action
        self.is_distributed = self.wrapped.is_distributed

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
        mirrored_observation = self._mirror_observation(observation)
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
            "mean": (original_action_dist["mean"] + self._mirror_action(mirrored_action_dist["mean"])) / 2,
            "std": (original_action_dist["std"] + abs(self._mirror_action(mirrored_action_dist["std"]))) / 2,
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
            action = (original_action + self._mirror_action(mirrored_action)) / 2
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
        mirrored_observation = self._mirror_observation(observation)
        mirrored_memory = self.wrapped.step_memory(mirrored_observation, memory=mirrored_memory, **kwargs)
        return None if memory is None else (memory, mirrored_memory)

    def reset_memory(self, memory: Memory, done: Slice | Tensor | None = None):
        if memory is None:
            return

        memory, mirrored_memory = memory
        self.wrapped.reset_memory(memory, done=done)
        self.wrapped.reset_memory(mirrored_memory, done=done)
