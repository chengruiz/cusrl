from io import StringIO
from typing import TypeVar

import numpy as np
import torch
from torch import nn

from cusrl.utils.config import CONFIG

__all__ = [
    "average_dict",
    "barrier",
    "enabled",
    "is_main_process",
    "gather_obj",
    "gather_stack",
    "gather_tensor",
    "local_rank",
    "make_distributed",
    "make_none_obj_list",
    "print_all",
    "print_once",
    "reduce_mean_",
    "reduce_mean_var_",
    "world_size",
]

_T = TypeVar("_T")


def average_dict(info_dict: dict[str, float]) -> dict[str, float]:
    if not CONFIG.distributed:
        return info_dict

    info_dict_list = gather_obj(info_dict)
    keys = {key for info in info_dict_list for key in info.keys()}
    result = {}
    for key in keys:
        values = [value for info in info_dict_list if (value := info.get(key, None)) is not None]
        if not values:
            continue
        result[key] = np.mean(values)
    return result


def barrier():
    if not CONFIG.distributed:
        return
    torch.distributed.barrier()


def enabled() -> bool:
    return CONFIG.distributed


def is_main_process() -> bool:
    """Checks if the current process is the main process."""
    return CONFIG.local_rank == 0


def gather_obj(obj: _T) -> list[_T]:
    if not CONFIG.distributed:
        return [obj]
    obj_list = [None for _ in range(CONFIG.world_size)]
    torch.distributed.all_gather_object(obj_list, obj)
    return obj_list


def gather_stack(tensor: torch.Tensor) -> torch.Tensor:
    if not CONFIG.distributed:
        return tensor.unsqueeze(0)

    gathered = tensor.new_empty(CONFIG.world_size, *tensor.shape)
    torch.distributed.all_gather_into_tensor(gathered, tensor)
    return gathered


def gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
    if not CONFIG.distributed:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(CONFIG.world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return tensor_list


def world_size() -> int:
    return CONFIG.world_size


def local_rank() -> int:
    return CONFIG.local_rank


def make_distributed(module, *, force=False):
    from cusrl.module.module import DistributedDataParallel

    if not CONFIG.distributed:
        raise RuntimeError("DistributedDataParallel is not enabled.")
    if isinstance(module, nn.parallel.DistributedDataParallel):
        return module
    if hasattr(module, "to_distributed") and not force:
        return module.to_distributed()
    return DistributedDataParallel(module)


def make_none_obj_list() -> list[object]:
    if not CONFIG.distributed:
        return []
    return [None for _ in range(CONFIG.world_size)]


def print_all(*args, **kwargs):
    if not CONFIG.distributed:
        print(*args, **kwargs)
        return
    buf = StringIO()
    print(*args, **kwargs, file=buf)

    output = make_none_obj_list()
    torch.distributed.all_gather_object(output, buf.getvalue())
    if CONFIG.local_rank == 0:
        for rank, out in enumerate(output):
            print(f"Rank {rank}: {out}", end="")


def print_once(*args, **kwargs):
    if not CONFIG.distributed or CONFIG.local_rank == 0:
        print(*args, **kwargs)


def reduce_mean_(tensor: torch.Tensor) -> torch.Tensor:
    """Reduces the tensor across all processes by averaging."""
    if not CONFIG.distributed:
        return tensor
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.AVG)
    return tensor


def reduce_mean_var_(mean: torch.Tensor, var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not CONFIG.distributed:
        return mean, var
    all_mean_var = gather_stack(torch.cat((mean, var), dim=0))
    all_means, all_vars = all_mean_var.chunk(2, -1)
    torch.mean(all_means, dim=0, out=mean)
    torch.mean(all_vars + (all_means - mean).square(), dim=0, out=var)
    return mean, var
