"""Utilities for initializing and coordinating distributed training helpers."""

from collections.abc import Iterable
from io import StringIO
from typing import Any, TypeVar, cast

import numpy as np
import torch

from cusrl.utils.config import CONFIG, configure_distributed

__all__ = [
    "average_dict",
    "barrier",
    "broadcast_parameters",
    "enabled",
    "is_main_process",
    "gather_obj",
    "gather_print",
    "gather_stack",
    "gather_tensor",
    "local_rank",
    "make_none_obj_list",
    "print_rank0",
    "rank",
    "reduce_gradients",
    "reduce_mean_",
    "reduce_mean_var_",
    "world_size",
]

_T = TypeVar("_T")


def average_dict(info_dict: dict[str, float]) -> dict[str, float]:
    """Gather dictionaries from all ranks and average values for each shared key."""
    if not configure_distributed():
        return info_dict

    info_dict_list = gather_obj(info_dict)
    keys = {key for info in info_dict_list for key in info.keys()}
    result: dict[str, float] = {}
    for key in keys:
        values = [value for info in info_dict_list if (value := info.get(key)) is not None]
        if not values:
            continue
        result[key] = float(np.mean(values))
    return result


def barrier():
    """Synchronize all ranks at a global barrier when distributed mode is enabled."""
    if not configure_distributed():
        return
    torch.distributed.barrier()


def broadcast_parameters(parameters: Iterable[torch.nn.Parameter]):
    """Broadcast parameter values from rank 0 to every other rank."""
    if not configure_distributed():
        return
    for param in parameters:
        torch.distributed.broadcast(param.data, src=0)


def enabled() -> bool:
    """Return whether distributed execution is enabled for the current process."""
    return CONFIG.distributed


def is_main_process() -> bool:
    """Checks if the current process is the main process."""
    return CONFIG.rank == 0


def gather_obj(obj: _T) -> list[_T]:
    """Gather a Python object from every rank into a list ordered by rank."""
    if not configure_distributed():
        return [obj]
    obj_list: list[Any] = [None for _ in range(CONFIG.world_size)]
    torch.distributed.all_gather_object(obj_list, obj)
    return cast(list[_T], obj_list)


def gather_print(*args, **kwargs):
    """Print once on rank 0 after collecting each rank's formatted output."""
    if not configure_distributed():
        print(*args, **kwargs)
        return
    buf = StringIO()
    print(*args, **kwargs, file=buf)

    output = make_none_obj_list()
    torch.distributed.all_gather_object(output, buf.getvalue())
    if CONFIG.rank == 0:
        rank_width = len(str(CONFIG.world_size - 1))
        for rank, out in enumerate(output):
            print(f"Rank {rank:0{rank_width}}: {out}", end="")


def gather_stack(tensor: torch.Tensor) -> torch.Tensor:
    """Gather matching tensors from all ranks and stack them along a new leading dimension."""
    if not configure_distributed():
        return tensor.unsqueeze(0)

    if torch.distributed.get_backend() == torch.distributed.Backend.GLOO:
        return torch.stack(gather_tensor(tensor), dim=0)
    gathered = tensor.new_empty(CONFIG.world_size, *tensor.shape)
    torch.distributed.all_gather_into_tensor(gathered, tensor)
    return gathered


def gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Gather a tensor from every rank into a Python list."""
    if not configure_distributed():
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(CONFIG.world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return tensor_list


def local_rank() -> int:
    """Return the local rank of the current process on its node."""
    return CONFIG.local_rank


def make_none_obj_list() -> list[object]:
    """Create an all-ranks placeholder list for object collective operations."""
    if not configure_distributed():
        return []
    return [None for _ in range(CONFIG.world_size)]


def print_rank0(*args, **kwargs):
    """Print only from rank 0."""
    if CONFIG.rank == 0:
        print(*args, **kwargs)


def rank() -> int:
    """Return the global rank of the current process."""
    return CONFIG.rank


def reduce_gradients(optimizer: torch.optim.Optimizer):
    """Reduces the gradients across all GPUs by averaging."""
    if not configure_distributed():
        return

    params = [param for group in optimizer.param_groups for param in group["params"] if param.grad is not None]
    if not params:
        return

    grads = torch.cat([param.grad.reshape(-1) for param in params])
    reduce_mean_(grads)

    offset = 0
    for param in params:
        numel = param.numel()
        param.grad.data.copy_(grads[offset : offset + numel].view_as(param.grad.data))
        offset += numel


def reduce_mean_(tensor: torch.Tensor) -> torch.Tensor:
    """Reduces the tensor across all processes by averaging."""
    if not configure_distributed():
        return tensor
    if torch.distributed.get_backend() == torch.distributed.Backend.GLOO:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return tensor.div_(CONFIG.world_size)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.AVG)
    return tensor


def reduce_mean_var_(mean: torch.Tensor, var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Average per-rank means and variances into the provided tensors in place."""
    if not configure_distributed():
        return mean, var
    all_mean_var = gather_stack(torch.cat((mean, var), dim=0))
    all_means, all_vars = all_mean_var.chunk(2, -1)
    torch.mean(all_means, dim=0, out=mean)
    torch.mean(all_vars + (all_means - mean).square(), dim=0, out=var)
    return mean, var


def world_size() -> int:
    """Return the total number of distributed processes."""
    return CONFIG.world_size
