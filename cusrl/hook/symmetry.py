from collections.abc import Sequence

import torch
from torch import Tensor, nn

from cusrl.module import Actor
from cusrl.template import ActorCritic, Hook
from cusrl.utils.typing import Memory, Slice

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
    """Implements a symmetry loss to facilitate symmetry in the action distribution.

    Described in "Learning Symmetric and Low-Energy Locomotion",
    https://dl.acm.org/doi/abs/10.1145/3197517.3201397

    Args:
        weight (float | None):
            Scaling factor for the symmetry loss. If None, symmetry loss is not applied.
    """

    MUTABLE_ATTRS = ["weight"]

    def __init__(self, weight: float | None):
        self.weight = weight
        self.mse_loss = nn.MSELoss()
        self.mirrored_actor_memory = None

    @torch.no_grad()
    def post_step(self, transition):
        actor = self.agent.actor
        mirrored_observation = self._mirror_observation(transition["observation"])
        transition["mirrored_actor_memory"] = self.mirrored_actor_memory
        self.mirrored_actor_memory = actor.step_memory(mirrored_observation, memory=self.mirrored_actor_memory)
        actor.reset_memory(self.mirrored_actor_memory, transition["done"])

    def objective(self, batch):
        if self.weight is None:
            return None

        actor = self.agent.actor
        (mirrored_action_mean, mirrored_action_std), _ = actor(
            self._mirror_observation(batch["observation"]),
            memory=batch.get("mirrored_actor_memory"),
            done=batch["done"],
        )

        mean_loss = self.mse_loss(batch["curr_action_mean"], self._mirror_action(mirrored_action_mean))
        std_loss = self.mse_loss(batch["curr_action_std"], self._mirror_action(mirrored_action_std))
        symmetry_loss = self.weight * (mean_loss + std_loss)
        self.agent.record(symmetry_loss=symmetry_loss)
        return symmetry_loss


class SymmetricDataAugmentation(SymmetryHook):
    """Augments training data by adding mirrored transitions to the batch.

    Described in "Symmetry Considerations for Learning Task Symmetric Robot Policies",
    https://ieeexplore.ieee.org/abstract/document/10611493

    This hook doubles the effective batch size by appending a mirrored version of
    each transition. For each transition (s, a, r, s'), it adds a corresponding
    mirrored transition (s_m, a_m, r, s'_m), where _m denotes a mirrored version.
    This encourages the learned policy to be symmetric.

    It also manages the recurrent state (memory) for the actor when processing
    mirrored observations, ensuring correct backpropagation through time for
    recurrent policies.
    """

    def __init__(self):
        self.mirrored_actor_memory = None

    @torch.no_grad()
    def post_step(self, transition):
        actor = self.agent.actor
        mirrored_observation = self._mirror_observation(transition["observation"])
        transition["mirrored_actor_memory"] = self.mirrored_actor_memory
        self.mirrored_actor_memory = actor.step_memory(mirrored_observation, memory=self.mirrored_actor_memory)
        actor.reset_memory(self.mirrored_actor_memory, transition["done"])

    def objective(self, batch):
        actor = self.agent.actor
        with self.agent.autocast():
            (mirrored_action_mean, mirrored_action_std), _ = actor(
                self._mirror_observation(batch["observation"]),
                memory=batch.get("mirrored_actor_memory"),
                done=batch["done"],
            )
            mirrored_action_logp = actor.calc_logp(
                mirrored_action_mean, mirrored_action_std, self._mirror_action(batch["action"])
            )
            mirrored_entropy = actor.calc_entropy(mirrored_action_mean, mirrored_action_std)
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
    the initialization phase, which ensures that the policy is strictly symmetric.
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
    ) -> tuple[tuple[Tensor, Tensor], Memory]:
        if memory is not None:
            memory, mirrored_memory = memory
        else:
            memory = mirrored_memory = None

        self.wrapped.intermediate_repr.clear()
        mirrored_observation = self._mirror_observation(observation)
        (mirrored_action_mean, mirrored_action_std), mirrored_memory = self.wrapped(
            mirrored_observation,
            memory=mirrored_memory,
            done=done,
            backbone_kwargs=backbone_kwargs,
            distribution_kwargs=distribution_kwargs,
        )
        mirrored_intermediate_repr = self.wrapped.intermediate_repr

        self.wrapped.intermediate_repr = {}
        (action_mean, action_std), memory = self.wrapped(
            observation,
            memory=memory,
            done=done,
            backbone_kwargs=backbone_kwargs,
            distribution_kwargs=distribution_kwargs,
        )

        self.intermediate_repr = self.wrapped.intermediate_repr
        self.intermediate_repr["action_mean"] = action_mean
        self.intermediate_repr["action_std"] = action_std
        self.intermediate_repr["mirrored.intermediate_repr"] = mirrored_intermediate_repr
        self.intermediate_repr["mirrored.action_mean"] = mirrored_action_mean
        self.intermediate_repr["mirrored.action_std"] = mirrored_action_std

        action_mean = (action_mean + self._mirror_action(mirrored_action_mean)) / 2
        action_std = (action_std + abs(self._mirror_action(mirrored_action_std))) / 2
        if memory is None:
            return (action_mean, action_std), None
        return (action_mean, action_std), (memory, mirrored_memory)

    def _explore_impl(
        self,
        observation: Tensor,
        memory: Memory = None,
        deterministic: bool = False,
        backbone_kwargs: dict | None = None,
        distribution_kwargs: dict | None = None,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], Memory]:
        (action_mean, action_std), memory = self(
            observation,
            memory=memory,
            backbone_kwargs=backbone_kwargs,
            distribution_kwargs=distribution_kwargs,
        )
        if deterministic:
            action = action_mean  # FIXME: directly using action_mean may not be correct
            logp = self.distribution.calc_logp(action_mean, action_std, action)
        else:
            action, logp = self.distribution.sample_from_dist(action_mean, action_std)
        return (action_mean, action_std), (action, logp), memory

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
