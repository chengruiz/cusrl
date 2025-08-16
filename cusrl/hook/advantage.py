from collections.abc import Sequence
from typing import Any, Literal, cast

import torch
from torch import Tensor

from cusrl.template import ActorCritic, Hook
from cusrl.utils import distributed

__all__ = ["AdvantageReduction", "AdvantageNormalization"]


class AdvantageReduction(Hook):
    """Reduces a multidimensional advantage tensor into a scalar.

    This hook reduces the advantage tensor along its last dimension. This is
    useful in multi-goal settings where the advantage is a vector.

    Args:
        reduction (Literal["sum", "mean"], optional):
            The reduction method to apply, either "sum" or "mean".
            Defaults to "sum".
        weight (Sequence[float] | None, optional):
            An optional sequence of weights to apply element-wise to the
            advantage tensor before reduction. Defaults to None.

    Raises:
        ValueError: If an invalid reduction method is provided.
    """

    def __init__(
        self,
        reduction: Literal["sum", "mean"] = "sum",
        weight: Sequence[float] | None = None,
    ):
        if reduction not in ("sum", "mean"):
            raise ValueError(f"Unknown reduction: '{reduction}'.")
        super().__init__()

        self.reduction = reduction

        # Mutable attributes
        self.weight: tuple[float, ...] | None
        self.register_mutable("weight", None if weight is None else tuple(weight))

        # Runtime attributes
        self._weight_tensor: Tensor | None

    def init(self):
        self._weight_tensor = None if self.weight is None else self.agent.to_tensor(self.weight)

    def objective(self, batch):
        advantage = cast(torch.Tensor, batch["advantage"])
        if self._weight_tensor is not None:
            advantage = advantage * self._weight_tensor
        if self.reduction == "sum":
            advantage = advantage.sum(-1, keepdim=True)
        elif self.reduction == "mean":
            advantage = advantage.mean(-1, keepdim=True)
        else:
            raise ValueError(f"Unknown reduction: '{self.reduction}'.")
        batch["advantage"] = advantage

    def update_attribute(self, name: str, value: Any):
        super().update_attribute(name, value)
        if name == "weight":
            if value is None:
                self.weight = self._weight_tensor = None
            else:
                self.weight = tuple(value)
                self._weight_tensor = self.agent.to_tensor(self.weight)


class AdvantageNormalization(Hook[ActorCritic]):
    """Normalizes advantages in actor-critic algorithms.

    This hook standardizes the advantages to have a mean of 0 and a standard
    deviation of 1. This can help stabilize training by preventing the scale of
    advantages from fluctuating wildly. Normalization can be configured to occur
    either once on the entire buffer before updates begin, or on each mini-batch
    during the objective calculation.

    The normalization correctly handles distributed training by averaging the
    mean and variance across all processes.

    Args:
        mini_batch_wise (bool, optional):
            If `True`, normalization is applied to each mini-batch. Defaults to
            `False`,
        synchronize (bool, optional):
            If `True`, the mean and variance are synchronized across all
            processes in distributed training. Defaults to `True`.
    """

    def __init__(self, mini_batch_wise: bool = False, synchronize: bool = True):
        super().__init__()
        self.mini_batch_wise = mini_batch_wise
        self.synchronize = synchronize

    def pre_update(self, buffer):
        if not self.mini_batch_wise:
            self.normalize_(cast(torch.Tensor, buffer["advantage"]))

    def objective(self, batch):
        if self.mini_batch_wise:
            self.normalize_(cast(torch.Tensor, batch["advantage"]))

    @torch.no_grad()
    def normalize_(self, advantage: Tensor):
        dims = tuple(range(advantage.ndim - 1))
        var, mean = torch.var_mean(advantage, dim=dims, correction=0)
        if self.synchronize:
            distributed.reduce_mean_var_(mean, var)
        std = (var + 1e-8).sqrt()
        advantage.sub_(mean).div_(std)
