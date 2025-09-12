import atexit
import os

import torch
from torch.distributed import GroupMember

__all__ = ["CONFIG", "device", "is_autocast_available"]


class Configurations:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self):
        self._cuda = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if self._cuda else "cpu")
        self._device_id = 0
        self._seed = None

        if "LOCAL_RANK" in os.environ:
            self._distributed = True
            self._rank = int(os.environ["RANK"])
            self._local_rank = int(os.environ["LOCAL_RANK"])
            self._world_size = int(os.environ["WORLD_SIZE"])
            if self._cuda:
                self._device = torch.device(f"cuda:{self._local_rank}")
                self._device_id = self._local_rank
            if GroupMember.WORLD is None:
                if self._rank == 0:
                    print(f"\033[1;32mInitializing distributed training with {self._world_size} processes.\033[0m")
                torch.distributed.init_process_group(
                    backend="nccl" if self._cuda else "gloo",
                    world_size=self._world_size,
                    rank=self._rank,
                    device_id=self._device,
                )
        else:
            self._distributed = False
            self._rank = 0
            self._local_rank = 0
            self._world_size = 1

        if self._cuda:
            torch.cuda.set_device(self._device)
        torch.set_float32_matmul_precision("high")

    @property
    def cuda(self) -> bool:
        return self._cuda

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def distributed(self) -> bool:
        return self._distributed

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def seed(self) -> int | None:
        return self._seed

    @seed.setter
    def seed(self, value: int | None):
        self._seed = value


def device(device: str | torch.device | None = None) -> torch.device:
    """Gets the specified device or default device if none specified."""
    if device is None:
        return CONFIG.device
    return torch.device(device)


def is_autocast_available() -> bool:
    return CONFIG.cuda and torch.amp.autocast_mode.is_autocast_available(CONFIG.device.type)


# Initialize global configuration
CONFIG = Configurations()


@atexit.register
def clean_distributed():
    if CONFIG.distributed and GroupMember.WORLD is not None:
        if CONFIG.rank == 0:
            print("\033[1;32mCleaning distributed training resources.\033[0m")
        torch.distributed.destroy_process_group()
