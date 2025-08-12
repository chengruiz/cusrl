from typing import cast

import torch

from cusrl.template import Hook

__all__ = ["RewardShaping"]


class RewardShaping(Hook):
    """Shapes the reward signal by scaling, shifting, and clamping.

    This hook modifies the reward in each transition by applying a linear
    transformation (reward = reward * scale + shift) and optionally clamping
    the result to a specified range.

    Args:
        scale (float, optional):
            The factor to scale the reward by. Defaults to 1.0.
        shift (float, optional):
            The constant to add to the reward. Defaults to 0.0.
        lower_bound (float | None):
            The minimum value to clamp the reward to. If None, no lower bound
            is applied. Defaults to None.
        upper_bound (float | None):
            The maximum value to clamp the reward to. If None, no upper bound
            is applied. Defaults to None.
    """

    def __init__(
        self,
        scale: float = 1.0,
        shift: float = 0.0,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ):
        super().__init__()
        self.scale = scale
        self.shift = shift
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def post_step(self, transition):
        reward = cast(torch.Tensor, transition["reward"])
        reward.mul_(self.scale).add_(self.shift)
        if self.lower_bound is not None or self.upper_bound is not None:
            reward.clamp_(min=self.lower_bound, max=self.upper_bound)
