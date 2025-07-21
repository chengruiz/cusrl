from typing import Any, overload

import numpy as np
import torch
from torch import nn

from cusrl.utils import distributed
from cusrl.utils.typing import ArrayType

__all__ = [
    "ExponentialMovingNormalizer",
    "RunningMeanStd",
    "mean_var_count",
    "synchronize_mean_var_count",
]


@overload
def mean_var_count(input: torch.Tensor, *, uncentered: bool = False) -> tuple[torch.Tensor, torch.Tensor, int]: ...
@overload
def mean_var_count(input: np.ndarray, *, uncentered: bool = False) -> tuple[np.ndarray, np.ndarray, int]: ...


def mean_var_count(input: ArrayType, *, uncentered: bool = False) -> tuple[ArrayType, ArrayType, int]:
    """Calculates mean, variance and count of the input array.

    Args:
        input: Input array with shape (N, D)
        uncentered: Whether to calculate uncentered variance

    Returns:
        Tuple of (mean, variance, count)
    """

    if isinstance(input, np.ndarray):
        mean, var, count = mean_var_count(torch.as_tensor(input), uncentered=uncentered)
        return mean.numpy(), var.numpy(), count

    if input.ndim < 2:
        raise ValueError("Input tensor must be at least 2-dimensional.")
    input = input.flatten(0, -2)
    count = int(input.size(0))
    if count == 0:
        mean = input.new_zeros(input.size(1))
        var = input.new_ones(input.size(1))
        return mean, var, count
    if uncentered:
        var = input.square().mean(dim=0)
        mean = torch.zeros_like(var)
    else:
        var, mean = torch.var_mean(input, dim=0, correction=0)
    return mean, var, count


def synchronize_mean_var_count(
    mean: torch.Tensor,
    var: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not distributed.enabled():
        return mean, var, count

    count_tensor = torch.tensor([count], dtype=mean.dtype, device=mean.device)
    # Only synchronize once for performance
    all_mean_var_count = distributed.gather_stack(torch.cat((mean, var, count_tensor), dim=0))

    dim = mean.size(0)
    all_means = all_mean_var_count[:, :dim]
    all_vars = all_mean_var_count[:, dim : 2 * dim]
    all_counts = all_mean_var_count[:, [2 * dim]]

    total_count = int(all_counts.sum().item())
    if total_count == 0:
        return mean, var, 0

    weights = all_counts / (total_count + 1e-8)
    total_mean = (all_means * weights).sum(dim=0)
    delta = all_means - total_mean
    total_var = torch.sum((all_vars + delta.square()) * weights, dim=0)

    return total_mean, total_var, total_count


def merge_mean_var_(
    old_mean: torch.Tensor,
    old_var: torch.Tensor,
    w_old: int | float,
    new_mean: torch.Tensor,
    new_var: torch.Tensor,
    w_new: int | float,
):
    w_sum = w_old + w_new
    if w_sum <= 0:
        raise ValueError(f"Weight sum must be positive, got {w_sum}.")
    w_old = w_old / w_sum
    w_new = w_new / w_sum
    delta = old_mean - new_mean
    old_mean.mul_(w_old).add_(new_mean * w_new)
    old_var.mul_(w_old).add_(new_var * w_new + delta.square() * (w_old * w_new))


class RunningMeanStd(nn.Module):
    """A class to track the running mean and standard deviation of a data stream.
    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm.
    """

    mean: torch.Tensor
    var: torch.Tensor
    std: torch.Tensor
    count: int

    def __init__(
        self,
        num_channels: int,
        *,
        clamp: float | None = 10.0,
        max_count: int | None = None,
        epsilon: float = 1e-8,
    ):
        if clamp is not None and clamp <= 0:
            raise ValueError("'clamp' should be None or positive.")
        if max_count is not None and max_count <= 0:
            raise ValueError("'max_count' should be None or positive.")
        self.clamp = clamp
        self.max_count = max_count
        self.epsilon = epsilon
        self.groups = []

        super().__init__()
        self.register_buffer("mean", torch.zeros(num_channels))
        self.register_buffer("var", torch.ones(num_channels))
        self.register_buffer("std", torch.ones(num_channels))
        self.count = 0
        self._is_synchronized = True
        self._synchronized_state = None

    def clear(self):
        self.mean.fill_(0.0)
        self.var.fill_(1.0)
        self.std.fill_(1.0)
        self.count = 0
        self.groups.clear()

    def register_stat_group(self, start_index: int, end_index: int):
        """Registers a group of indices where statistics will be shared/averaged during updates."""
        self.groups.append((start_index, end_index))

    def update(
        self,
        input: torch.Tensor,
        *,
        uncentered: bool = False,
        synchronize: bool = True,
    ):
        """Updates statistics with new data.

        Args:
            input: Input tensor
            uncentered: Whether to calculate uncentered variance
            synchronize: Whether to synchronize across devices
        """
        self.update_from_stats(
            *mean_var_count(input, uncentered=uncentered),
            synchronize=synchronize,
        )

    @torch.no_grad()
    def update_from_stats(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: int,
        *,
        synchronize: bool = True,
    ):
        if synchronize:
            self.synchronize()
            batch_mean, batch_var, batch_count = synchronize_mean_var_count(batch_mean, batch_var, batch_count)
        if batch_count == 0:
            return
        self._average_intra_group(batch_mean, batch_var)
        self._update_mean_var(batch_mean, batch_var, batch_count)
        self.std.copy_(torch.sqrt(self.var + self.epsilon))
        self.count += batch_count
        self._is_synchronized = synchronize
        if self._is_synchronized:
            if self.max_count is not None and self.count > self.max_count:
                self.count = self.max_count
            self._synchronized_state = (self.mean.clone(), self.var.clone(), self.count)

    def synchronize(self):
        if self._is_synchronized or not distributed.enabled():
            return
        if self._synchronized_state is None:
            total_mean, total_var, total_count = synchronize_mean_var_count(self.mean, self.var, self.count)
        else:
            sync_mean, sync_var, sync_count = self._synchronized_state
            merge_mean_var_(self.mean, self.var, self.count, sync_mean, sync_var, -sync_count)
            patch = synchronize_mean_var_count(self.mean, self.var, self.count - sync_count)
            merge_mean_var_(sync_mean, sync_var, sync_count, *patch)
            total_mean, total_var, total_count = sync_mean, sync_var, sync_count + patch[2]

        self.mean.copy_(total_mean)
        self.var.copy_(total_var)
        self.std.copy_(torch.sqrt(total_var + self.epsilon))
        self.count = total_count
        if self.max_count is not None and self.count > self.max_count:
            self.count = self.max_count
        self._is_synchronized = True
        self._synchronized_state = (total_mean, total_var, self.count)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.normalize(input)

    def normalize(self, input: torch.Tensor) -> torch.Tensor:
        """Normalizes the given values."""
        output = (input - self.mean) / self.std
        if self.clamp is not None:
            output = output.clamp(-self.clamp, self.clamp)
        return output.to(input.dtype)

    def normalize_(self, input: torch.Tensor) -> torch.Tensor:
        """Inplace version of `normalize`."""
        input.sub_(self.mean).div_(self.std)
        if self.clamp is not None:
            input.clamp_(-self.clamp, self.clamp)
        return input

    def unnormalize(self, input: torch.Tensor) -> torch.Tensor:
        """Unnormalizes the given values."""
        return (input * self.std + self.mean).to(input.dtype)

    def unnormalize_(self, input: torch.Tensor) -> torch.Tensor:
        """Inplace version of `unnormalize`."""
        return input.mul_(self.std).add_(self.mean)

    def _update_mean_var(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int):
        merge_mean_var_(self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def _average_intra_group(self, batch_mean: torch.Tensor, batch_var: torch.Tensor):
        """Collapse the statistics within dimensions in registered groups."""
        for start, end in self.groups:
            group_mean = batch_mean[start:end].mean()
            group_squared_mean = batch_mean[start:end].square().mean()
            group_var = batch_var[start:end].mean() - group_mean.square() + group_squared_mean
            batch_mean[start:end] = group_mean
            batch_var[start:end] = group_var

    def get_extra_state(self) -> Any:
        return torch.tensor(self.count, dtype=torch.int64)

    def set_extra_state(self, state: Any):
        if state < 0:
            raise ValueError("State must be non-negative.")
        self.count = int(state.item() if isinstance(state, torch.Tensor) else state)
        self._synchronized_state = (self.mean.clone(), self.var.clone(), self.count)


class ExponentialMovingNormalizer(RunningMeanStd):
    def __init__(
        self,
        num_channels: int,
        alpha: float,
        *,
        warmup: bool = False,
        clamp: float | None = 10.0,
        epsilon: float = 1e-8,
    ):
        if not (0 < alpha <= 1):
            raise ValueError("'alpha' must be in the range (0, 1].")
        super().__init__(num_channels, clamp=clamp, epsilon=epsilon)
        self.alpha = alpha
        self.warmup = warmup

    def _update_mean_var(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int):
        wb = self.alpha
        if self.warmup:
            wb = max(batch_count / (batch_count + self.count), wb)
        merge_mean_var_(self.mean, self.var, 1.0 - wb, batch_mean, batch_var, wb)
