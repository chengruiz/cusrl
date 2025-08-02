from cusrl.template import ActorCritic, Hook

__all__ = ["OnPolicyPreparation"]


class OnPolicyPreparation(Hook[ActorCritic]):
    """Prepares data for on-policy reinforcement learning algorithms.

    This hook processes a batch of data to compute current policy statistics,
    including action mean, standard deviation, log probability, entropy, and
    probability ratio. Optionally computes KL divergence if enabled.

    Args:
        calculate_kl_divergence (bool): If True, computes the KL divergence
        between the old and current policy distributions.
    """

    def __init__(self, calculate_kl_divergence: bool = False):
        super().__init__()
        self.calculate_kl_divergence = calculate_kl_divergence

    def objective(self, batch):
        actor = self.agent.actor

        with self.agent.autocast():
            action_dist, _ = actor(
                batch["observation"],
                memory=batch.get("actor_memory"),
                done=batch["done"],
            )
            action_logp = actor.compute_logp(action_dist, batch["action"])
            entropy = actor.compute_entropy(action_dist)
            logp_diff = action_logp - batch["action_logp"]
        self.agent.record(ratio=logp_diff.abs(), entropy=entropy)

        batch["curr_action_dist"] = action_dist
        batch["curr_action_logp"] = action_logp
        batch["curr_entropy"] = entropy
        batch["action_logp_diff"] = logp_diff
        batch["action_prob_ratio"] = logp_diff.exp()
        if self.calculate_kl_divergence:
            batch["kl_divergence"] = actor.compute_kl_div(batch["action_dist"], action_dist)
