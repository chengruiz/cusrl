from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from cusrl.nn.utils.normalization import mean_var_count, merge_mean_var_, synchronize_mean_var_count
from cusrl.utils import distributed
from cusrl.utils.typing import Slice

__all__ = [
    "ExponentialMovingNormalizer",
    "RunningMeanStd",
]


class RunningMeanStd(nn.Module):
    """Tracks the running mean and standard deviation of a datastream.
    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm.

    This module is used to normalize data based on statistics collected from
    previously seen data. It supports distributed training by synchronizing
    statistics across multiple processes. It also allows for grouping channels
    to share the same running statistics, which can be useful for features that
    should be normalized together.

    Args:
        num_channels (int):
            The number of channels of the input data.
        groups (Iterable[Slice], optional):
            Indices of channel dimensions that share the same statistics.
            Defaults to ``()``.
        excluded_indices (Slice | None, optional):
            Indices of channel dimensions that are excluded from normalization.
            Defaults to ``None``.
        clamp (float | None, optional):
            If not ``None``, the normalized output will be clamped to the range
            ``[-clamp, clamp]``. Defaults to ``10.0``.
        max_count (int | None, optional):
            If not ``None``, the count will be capped to this value. Defaults to
            ``None``.
        epsilon (float, optional):
            A small value added to the variance to avoid division by zero.
            Defaults to ``1e-8``.

    Attributes:
        mean (Tensor):
            The running mean, shape :math:`(C,)`, where :math:`C` is the number
            of channels.
        var (Tensor):
            The running variance, shape :math:`(C,)`.
        std (Tensor):
            The running standard deviation, shape :math:`(C,)`.
        count (int):
            The number of samples seen so far.

    Raises:
        ValueError: If `clamp` or `max_count` is non-positive.
        ValueError: If `groups` contain overlapping indices.
    """

    def __init__(
        self,
        num_channels: int,
        *,
        groups: Iterable[Slice] = (),
        excluded_indices: Slice | None = None,
        clamp: float | None = 10.0,
        max_count: int | None = None,
        epsilon: float = 1e-8,
    ):
        if clamp is not None and clamp <= 0:
            raise ValueError("'clamp' must be None or a positive value")
        if max_count is not None and max_count <= 0:
            raise ValueError("'max_count' must be None or a positive value")
        self.groups = tuple(groups)
        self.excluded_indices = excluded_indices
        self.clamp = clamp
        self.max_count = max_count
        self.epsilon = epsilon

        dummy_input = torch.zeros(num_channels, dtype=torch.int64)
        for indices in self.groups:
            dummy_input[indices,] += 1
        if torch.any(dummy_input > 1):
            raise ValueError("Indices in 'groups' must not overlap")

        super().__init__()
        self.mean: Tensor
        self.var: Tensor
        self.std: Tensor
        self.register_buffer("mean", torch.zeros(num_channels))
        self.register_buffer("var", torch.ones(num_channels))
        self.register_buffer("std", torch.ones(num_channels))
        self.count: int = 0

        self._is_synchronized = True
        self._synchronized_state: tuple[Tensor, Tensor, int] | None = None

    def clear(self):
        self.mean.fill_(0.0)
        self.var.fill_(1.0)
        self.std.fill_(1.0)
        self.count = 0

    def update(
        self,
        input: Tensor,
        *,
        uncentered: bool = False,
        synchronize: bool = True,
    ):
        """Updates statistics with new data.

        Args:
            input (Tensor):
                Input tensor.
            uncentered (bool, optional):
                Whether to calculate uncentered variance. Defaults to ``False``.
            synchronize (bool, optional):
                Whether to synchronize across devices. Defaults to ``True``.
        """
        self.update_from_stats(
            *mean_var_count(input, uncentered=uncentered),
            synchronize=synchronize,
        )

    @torch.no_grad()
    def update_from_stats(
        self,
        batch_mean: Tensor,
        batch_var: Tensor,
        batch_count: int,
        *,
        synchronize: bool = True,
    ):
        if synchronize:
            self.synchronize()
            batch_mean, batch_var, batch_count = synchronize_mean_var_count(batch_mean, batch_var, batch_count)
        if batch_count == 0:
            return
        self._process_mean_var(batch_mean, batch_var)
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

    def forward(self, input: Tensor) -> Tensor:
        return self.normalize(input)

    def normalize(self, input: Tensor) -> Tensor:
        """Normalizes the given values."""
        output = (input - self.mean) / self.std
        if self.clamp is not None:
            output = output.clamp(-self.clamp, self.clamp)
        return output.type_as(input)

    def normalize_(self, input: Tensor) -> Tensor:
        """Inplace version of `normalize`."""
        input.sub_(self.mean).div_(self.std)
        if self.clamp is not None:
            input.clamp_(-self.clamp, self.clamp)
        return input

    def unnormalize(self, input: Tensor) -> Tensor:
        """Unnormalizes the given values."""
        return (input * self.std + self.mean).type_as(input)

    def unnormalize_(self, input: Tensor) -> Tensor:
        """Inplace version of `unnormalize`."""
        return input.mul_(self.std).add_(self.mean)

    def _process_mean_var(self, batch_mean: Tensor, batch_var: Tensor):
        if self.excluded_indices is not None:
            batch_mean[self.excluded_indices,] = 0.0
            batch_var[self.excluded_indices,] = 1.0
        for indices in self.groups:
            group_mean = batch_mean[indices,].mean()
            group_squared_mean = batch_mean[indices,].square().mean()
            group_var = batch_var[indices,].mean() - group_mean.square() + group_squared_mean
            batch_mean[indices,] = group_mean
            batch_var[indices,] = group_var

    def _update_mean_var(self, batch_mean: Tensor, batch_var: Tensor, batch_count: int):
        merge_mean_var_(self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def get_extra_state(self) -> Any:
        return torch.tensor(self.count, dtype=torch.int64)

    def set_extra_state(self, state: Any):
        if state < 0:
            raise ValueError("The normalizer state count must be non-negative")
        self.count = int(state.item() if isinstance(state, Tensor) else state)
        self._synchronized_state = (self.mean.clone(), self.var.clone(), self.count)


class ExponentialMovingNormalizer(RunningMeanStd):
    def __init__(
        self,
        num_channels: int,
        alpha: float,
        *,
        groups: Iterable[Slice] = (),
        excluded_indices: Slice | None = None,
        warmup: bool = False,
        clamp: float | None = 10.0,
        epsilon: float = 1e-8,
    ):
        if not (0 < alpha <= 1):
            raise ValueError("'alpha' must be in the range (0, 1]")
        super().__init__(
            num_channels,
            groups=groups,
            excluded_indices=excluded_indices,
            clamp=clamp,
            epsilon=epsilon,
        )
        self.alpha = alpha
        self.warmup = warmup

    def _update_mean_var(self, batch_mean: Tensor, batch_var: Tensor, batch_count: int):
        wb = self.alpha
        if self.warmup:
            wb = max(batch_count / (batch_count + self.count), wb)
        merge_mean_var_(self.mean, self.var, 1.0 - wb, batch_mean, batch_var, wb)
