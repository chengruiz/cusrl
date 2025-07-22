from typing import Any

import torch

from cusrl.template import Hook

__all__ = ["PpoSurrogateLoss", "EntropyLoss"]


def _ppo_surrogate_loss(
    advantage: torch.Tensor,
    prob_ratio: torch.Tensor,
    clip_ratio: float,
):
    return -torch.min(
        advantage * prob_ratio,
        advantage * prob_ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio),
    ).mean()


class PpoSurrogateLoss(Hook):
    """Calculates the PPO surrogate loss.

    This hook implements the clipped surrogate objective function from the Proximal
    Policy Optimization (PPO) algorithm. It computes the loss based on the
    advantage estimates and the ratio of the new policy's action probabilities
    to the old policy's action probabilities. The objective is clipped to
    discourage large policy updates.

    Args:
        clip_ratio (float, optional):
            The clipping parameter for the PPO surrogate loss. It determines
            the range `[1 - clip_ratio, 1 + clip_ratio]` within which the probability
            ratio is clipped. Defaults to 0.2.
    """

    MUTABLE_ATTRS = ["clip_ratio"]

    def __init__(self, clip_ratio: float = 0.2):
        super().__init__()
        self.clip_ratio = clip_ratio

    def objective(self, batch: dict[str, Any]):
        if (advantage := batch["advantage"]).size(-1) != 1:
            raise ValueError(f"Expected advantage to have shape [..., 1], got {advantage.shape}.")
        with self.agent.autocast():
            surrogate_loss = _ppo_surrogate_loss(
                advantage,
                batch["action_prob_ratio"],
                self.clip_ratio,
            )
        self.agent.record(surrogate_loss=surrogate_loss)
        return surrogate_loss


class EntropyLoss(Hook):
    """Calculates the entropy loss to encourage exploration.

    This hook implements the entropy bonus, a common component in policy gradient
    algorithms like PPO. By adding the negative entropy of the policy's action
    distribution to the main objective, it encourages the policy to maintain
    high entropy, thus promoting exploration and preventing premature convergence
    to a suboptimal deterministic policy.

    Args:
        weight (float, optional):
            The coefficient for the entropy loss term. A larger value results in a
            stronger incentive for exploration. Defaults to 0.01.
    """

    MUTABLE_ATTRS = ["weight"]

    def __init__(self, weight: float = 0.01):
        self.weight = weight

    def objective(self, batch: dict[str, Any]):
        return -self.weight * batch["curr_entropy"].mean()
