from typing import Any

from cusrl.template import ActorCritic, Hook

__all__ = ["OnPolicyPreparation"]


class OnPolicyPreparation(Hook[ActorCritic]):
    def __init__(self, calculate_kl_divergence: bool = False):
        self.calculate_kl_divergence = calculate_kl_divergence

    def objective(self, batch: dict[str, Any]):
        actor = self.agent.actor

        with self.agent.autocast():
            (action_mean, action_std), _ = actor(
                batch["observation"],
                memory=batch.get("actor_memory"),
                done=batch["done"],
            )
            action_logp = actor.calc_logp(action_mean, action_std, batch["action"])
            entropy = actor.calc_entropy(action_mean, action_std)
            logp_diff = action_logp - batch["action_logp"]
        self.agent.record(ratio=logp_diff.abs(), entropy=entropy)

        batch["curr_action_mean"] = action_mean
        batch["curr_action_std"] = action_std
        batch["curr_action_logp"] = action_logp
        batch["curr_entropy"] = entropy
        batch["action_logp_diff"] = logp_diff
        batch["action_prob_ratio"] = logp_diff.exp()
        if self.calculate_kl_divergence:
            batch["kl_divergence"] = actor.distribution.calc_kl_div(
                batch["action_mean"], batch["action_std"], action_mean, action_std
            )
