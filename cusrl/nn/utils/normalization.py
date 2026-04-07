from typing import overload

import numpy as np
import torch
from torch import Tensor

from cusrl.utils import distributed
from cusrl.utils.typing import ArrayType

__all__ = [
    "mean_var_count",
    "merge_mean_var_",
    "synchronize_mean_var_count",
]


@overload
def mean_var_count(input: Tensor, *, uncentered: bool = False) -> tuple[Tensor, Tensor, int]: ...
@overload
def mean_var_count(input: np.ndarray, *, uncentered: bool = False) -> tuple[np.ndarray, np.ndarray, int]: ...


def mean_var_count(input: ArrayType, *, uncentered: bool = False) -> tuple[ArrayType, ArrayType, int]:
    """Calculates mean, variance and count of the input array.

    Args:
        input (np.ndarray | Tensor):
            Input array of shape :math:`(N, C)`.
        uncentered (bool, optional):
            Whether to calculate uncentered variance. Defaults to False.

    Returns:
        - mean (np.ndarray | Tensor):
            The mean of the input array.
        - var (np.ndarray | Tensor):
            The variance of the input array.
        - count (int):
            The number of samples in the input array.
    """

    if isinstance(input, np.ndarray):
        mean, var, count = mean_var_count(torch.as_tensor(input), uncentered=uncentered)
        return mean.numpy(), var.numpy(), count

    if input.ndim < 2:
        raise ValueError("Input tensor must be at least 2-dimensional")
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


def synchronize_mean_var_count(mean: Tensor, var: Tensor, count: int) -> tuple[Tensor, Tensor, int]:
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
    old_mean: Tensor,
    old_var: Tensor,
    w_old: int | float,
    new_mean: Tensor,
    new_var: Tensor,
    w_new: int | float,
):
    w_sum = w_old + w_new
    if w_sum <= 0:
        raise ValueError(f"Weight sum must be positive; got {w_sum}")
    w_old = w_old / w_sum
    w_new = w_new / w_sum
    delta = new_mean - old_mean
    old_mean.add_(delta * w_new)
    old_var.add_((new_var - old_var) * w_new + delta.square() * (w_old * w_new))
