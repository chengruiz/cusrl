from collections.abc import Callable
from typing import cast

import torch

from cusrl.template import ActorCritic, Hook, Sampler

__all__ = [
    "OnPolicyBufferCapacitySchedule",
    "OnPolicyPreparation",
    "OnPolicyStatistics",
]


class OnPolicyBufferCapacitySchedule(Hook[ActorCritic]):
    """Schedules the capacity of a rollout buffer for an on-policy agent.

    This hook uses a user-provided schedule function that maps the current
    training iteration to the desired number of environment steps per update
    (i.e., rollout length) and resizes the agent's buffer accordingly.

    Args:
        schedule (Callable[[int], int]):
            Function that takes the current iteration index and returns the
            desired buffer capacity. This typically controls the rollout length
            before each iteration.
    """

    def __init__(self, schedule: Callable[[int], int]):
        super().__init__()
        self.schedule = schedule

    def apply_schedule(self, iteration: int):
        capacity = self.schedule(iteration)
        self.agent.num_steps_per_update = capacity
        self.agent.resize_buffer(capacity)


class OnPolicyPreparation(Hook[ActorCritic]):
    """Evaluates the current policy before on-policy update.

    This hook processes a batch of data to compute current policy statistics,
    including action distribution parameters, log probability, entropy, and
    probability ratio. Optionally computes KL divergence if enabled.

    Args:
        calculate_kl_divergence (bool):
            If True, computes the KL divergence between the old and current
            policy distributions.
    """

    def __init__(self, calculate_kl_divergence: bool = False):
        super().__init__()
        self.calculate_kl_divergence = calculate_kl_divergence

    def objective(self, batch):
        actor = self.agent.actor

        with self.agent.autocast():
            action_dist, _ = actor(batch["observation"], memory=batch.get("actor_memory"), done=batch["done"])
            action_logp = actor.compute_logp(action_dist, batch["action"])
            entropy = actor.compute_entropy(action_dist)
            logp_ratio = action_logp - cast(torch.Tensor, batch["action_logp"])
        self.agent.record(ratio=logp_ratio.abs(), entropy=entropy)

        batch["curr_action_dist"] = action_dist
        batch["curr_action_logp"] = action_logp
        batch["curr_entropy"] = entropy
        batch["action_logp_ratio"] = logp_ratio
        batch["action_prob_ratio"] = logp_ratio.exp()
        if self.calculate_kl_divergence:
            batch["kl_divergence"] = actor.compute_kl_div(batch["action_dist"], action_dist)


class OnPolicyStatistics(Hook[ActorCritic]):
    """Calculates and records on-policy statistics after update phase.

    Specifically, it records:
    - `kl_divergence`: The Kullback-Leibler divergence between the policy
        before and after the update.
    - `action_std`: The standard deviation of the action distribution from the
        updated policy.

    Args:
        sampler (Sampler | None, optional):
            The sampler used to sample batches from the agent's buffer. If None,
            a default `Sampler()` is used. Defaults to None.
    """

    def __init__(self, sampler: Sampler | None = None):
        super().__init__()
        self.sampler = sampler if sampler is not None else Sampler()

    @torch.no_grad()
    def post_update(self):
        actor = self.agent.actor
        for batch in self.sampler(self.agent.buffer):
            with self.agent.autocast():
                action_dist, _ = actor(batch["observation"], memory=batch.get("actor_memory"), done=batch["done"])
            self.agent.record(kl_divergence=actor.compute_kl_div(batch["action_dist"], action_dist))
            if "std" in action_dist:
                self.agent.record(action_std=action_dist["std"])
