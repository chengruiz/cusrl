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
        self.sampler = sampler if sampler is not None else Sampler()

    @torch.inference_mode()
    def post_update(self):
        agent = self.agent
        for batch in self.sampler(agent.buffer):
            with agent.autocast():
                (action_mean, action_std), _ = agent.actor(
                    batch["observation"],
                    memory=batch.get("actor_memory"),
                    done=batch["done"],
                )

            agent.record(
                kl_divergence=agent.actor.distribution.calc_kl_div(
                    batch["action_mean"], batch["action_std"], action_mean, action_std
                ),
                action_std=action_std,
            )
