import torch

from cusrl.template import ActorCritic, Hook, Sampler

__all__ = ["OnPolicyStatistics"]


class OnPolicyStatistics(Hook[ActorCritic]):
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
