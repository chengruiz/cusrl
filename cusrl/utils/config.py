import atexit
from datetime import datetime
import os
import re
import socket

import torch
from torch.distributed import GroupMember

__all__ = ["CONFIG", "configure_distributed", "device", "is_autocast_available"]


def _normalize_identifier(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    value = value.strip("._-")
    return value or "unknown"


class Configurations:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self):
        self._cuda = torch.cuda.is_available()
        self._seed = None
        self.device = torch.device("cuda:0" if self._cuda else "cpu")

        if "LOCAL_RANK" in os.environ:
            self._distributed = True
            self._rank = int(os.environ["RANK"])
            self._local_rank = int(os.environ["LOCAL_RANK"])
            self._world_size = int(os.environ["WORLD_SIZE"])
            self._local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            if self._cuda:
                self.device = torch.device(f"cuda:{self._local_rank}")

            # Set unique directories for each process to avoid conflicts
            idenfier = self._get_distributed_identifier()
            torchinductor_root = os.getenv("TORCHINDUCTOR_CACHE_DIR", f"/tmp/cache/torchinductor/{torch.__version__}")
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(torchinductor_root, idenfier)
            if (triton_root := os.getenv("TRITON_CACHE_DIR")) is None:
                os.environ["TRITON_CACHE_DIR"] = os.path.join(os.environ["TORCHINDUCTOR_CACHE_DIR"], "triton")
            else:
                os.environ["TRITON_CACHE_DIR"] = os.path.join(triton_root, idenfier)
            try:
                import warp as wp

                warp_root = os.getenv("WARP_CACHE_PATH", f"/tmp/cache/warp/{wp.__version__}")
                os.environ["WARP_CACHE_PATH"] = os.path.join(warp_root, idenfier)
            except ImportError:
                pass
        else:
            self._distributed = False
            self._rank = 0
            self._local_rank = 0
            self._world_size = 1
            self._local_world_size = 1

        try:
            import flash_attn  # noqa: F401

            self._flash_attention_found = True
        except ImportError:
            self._flash_attention_found = False
        self._flash_attention_enabled = self._flash_attention_found

    @property
    def cuda(self) -> bool:
        return self._cuda

    @property
    def seed(self) -> int | None:
        return self._seed

    @seed.setter
    def seed(self, value: int | None):
        self._seed = value

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, value: str | torch.device):
        self._device = torch.device(value)

    def set_device(self, value: str | torch.device):
        self.device = value

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
    def local_world_size(self) -> int:
        return self._local_world_size

    @property
    def flash_attention_enabled(self) -> bool:
        return self._flash_attention_enabled

    @flash_attention_enabled.setter
    def _(self, value: bool):
        self.enable_flash_attention(value)

    def enable_flash_attention(self, enabled: bool = True):
        if enabled and not self._flash_attention_found:
            raise RuntimeError("Cannot enable 'flash_attn' because it is not installed")
        self._flash_attention_enabled = enabled

    def _get_distributed_identifier(self) -> str:
        if (job_id := os.getenv("TORCHELASTIC_RUN_ID")) == "none":
            fallback_parts = []
            if (master_addr := os.getenv("MASTER_ADDR")) is not None:
                fallback_parts.append(f"master_{_normalize_identifier(master_addr)}")
            if (master_port := os.getenv("MASTER_PORT")) is not None:
                fallback_parts.append(f"port_{_normalize_identifier(master_port)}")
            fallback_parts.append(datetime.now().strftime("%Y%m%d%H%M%S"))
            fallback_parts.append(f"pid_{os.getpid()}")
            job_id = "__".join(fallback_parts)
        host = _normalize_identifier(socket.gethostname())
        return os.path.join(f"host_{host}", f"job_{job_id}", f"rank{self._rank}")


def device(device: str | torch.device | None = None) -> torch.device:
    """Gets the specified device or default device if none specified."""
    if device is None:
        return CONFIG.device
    return torch.device(device)


def is_autocast_available() -> bool:
    return CONFIG.cuda and torch.amp.autocast_mode.is_autocast_available(CONFIG.device.type)


def configure_distributed(
    backend: str | None = None,
    **kwargs,
) -> bool:
    if not CONFIG.distributed:
        return False
    if backend is None:
        if hasattr(torch.distributed, "get_default_backend_for_device"):
            backend = torch.distributed.get_default_backend_for_device(CONFIG.device)
        else:
            backend = "nccl" if CONFIG.device.type == "cuda" else "gloo"

    if GroupMember.WORLD is None:
        from cusrl.utils.distributed import print_rank0

        print_rank0(
            f"\033[1;32mInitializing distributed training (backend={backend}, world_size={CONFIG.world_size}).\033[0m"
        )
        torch.distributed.init_process_group(
            backend=backend,
            world_size=CONFIG.world_size,
            rank=CONFIG.rank,
            **kwargs,
        )
        if CONFIG.device.type == "cuda":
            torch.cuda.set_device(CONFIG.device)
    return True


# Initialize global configuration
CONFIG = Configurations()


@atexit.register
def clean_distributed():
    if CONFIG.distributed and GroupMember.WORLD is not None:
        if CONFIG.rank == 0:
            print("\033[1;32mCleaning distributed training resources.\033[0m")
        torch.distributed.destroy_process_group()
