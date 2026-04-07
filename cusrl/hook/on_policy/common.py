from typing import cast

import torch

from cusrl.template import ActorCritic, Hook

__all__ = [
    "OnPolicyPreparation",
]


class OnPolicyPreparation(Hook[ActorCritic]):
    """Evaluates the current policy before on-policy update.

    This hook processes a batch of data to compute current policy statistics,
    including action distribution parameters, log probability, entropy, and
    probability ratio. Optionally computes KL divergence if enabled.

    Args:
        calculate_kl_divergence (bool, optional):
            If ``True``, computes the KL divergence between the old and current
            policy distributions. Defaults to ``False``.
    """

    def __init__(self, calculate_kl_divergence: bool = False):
        super().__init__(training_only=True)
        self.calculate_kl_divergence = calculate_kl_divergence

    def objective(self, metadata, batch):
        actor = self.agent.actor

        action_dist, _ = actor(batch["observation"], memory=batch.get("actor_memory"), done=batch["done"])
        action_logp = actor.compute_logp(action_dist, batch["action"])
        entropy = actor.compute_entropy(action_dist)
        logp_ratio = action_logp - cast(torch.Tensor, batch["action_logp"])

        batch["curr_action_dist"] = action_dist
        batch["curr_action_logp"] = action_logp
        batch["curr_entropy"] = entropy
        batch["action_logp_ratio"] = logp_ratio
        batch["action_prob_ratio"] = logp_ratio.exp()
        if self.calculate_kl_divergence:
            batch["kl_divergence"] = actor.compute_kl_div(batch["action_dist"], action_dist)

    def post_objective(self, batch):
        self.agent.record(
            ratio=batch["action_logp_ratio"].abs(),
            entropy=batch["curr_entropy"],
        )
