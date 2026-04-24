import copy

import torch

from cusrl.template import ActorCritic, Hook
from cusrl.utils import distributed

__all__ = ["UnsafeUpdateRollback"]


class UnsafeUpdateRollback(Hook[ActorCritic]):
    """Rejects an on-policy update when the post-update KL is too large.

    The hook snapshots the full agent checkpoint state before an update and
    restores it after the update if the recorded ``kl_divergence`` metric is
    greater than ``max_kl_divergence``. Place this hook after
    :class:`OnPolicyStatistics` so the KL metric is available.

    Args:
        max_kl_divergence:
            Maximum accepted KL divergence for one update.
    """

    def __init__(self, max_kl_divergence: float):
        if max_kl_divergence <= 0:
            raise ValueError("'max_kl_divergence' must be positive")
        super().__init__(training_only=True)
        self.max_kl_divergence = max_kl_divergence
        self.register_mutable("max_kl_divergence")
        self._checkpoint: dict | None = None

    def pre_update(self, buffer):
        self._checkpoint = copy.deepcopy(self.agent.state_dict())

    @torch.no_grad()
    def post_update(self):
        if self._checkpoint is None:
            return

        kl_divergence = self.agent.metrics["kl_divergence"].mean.clone()
        distributed.reduce_mean_(kl_divergence)
        rejected = kl_divergence.item() > self.max_kl_divergence
        if rejected:
            self.agent.load_state_dict(self._checkpoint)
            self.agent.record(rollback_kl_divergence=kl_divergence, update_rejected=1.0)
        else:
            self.agent.record(update_rejected=0.0)
        self._checkpoint = None
