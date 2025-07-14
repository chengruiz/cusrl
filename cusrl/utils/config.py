import atexit
import os

import torch
from torch.distributed import GroupMember

__all__ = ["CONFIG", "device", "is_autocast_available"]


class Configurations:
    cuda: bool
    device: torch.device
    device_id: int
    distributed: bool
    local_rank: int
    world_size: int
    seed: int | None = None
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        self.device_id = 0

        if "LOCAL_RANK" in os.environ:
            self.distributed = True
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            if self.cuda:
                self.device_id = self.local_rank
                self.device = torch.device(f"cuda:{self.device_id}")
                torch.cuda.set_device(self.device_id)
            if GroupMember.WORLD is None:
                if self.local_rank == 0:
                    print(f"\033[1;32mInitializing distributed training with {self.world_size} processes.\033[0m")
                torch.distributed.init_process_group(
                    backend="nccl" if self.cuda else "gloo",
                    world_size=self.world_size,
                    rank=self.local_rank,
                    device_id=self.device,
                )
        else:
            self.distributed = False
            self.local_rank = 0
            self.world_size = 1

        torch.set_float32_matmul_precision("high")


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
        if CONFIG.local_rank == 0:
            print("\033[1;32mCleaning distributed training resources.\033[0m")
        torch.distributed.destroy_process_group()
