import torch

from cusrl.template import ActorCritic, Hook, Sampler

__all__ = ["OnPolicyStatistics"]


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
