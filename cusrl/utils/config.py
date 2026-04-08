import atexit
import os
import re
import socket
from datetime import datetime

import torch
from torch.distributed import GroupMember

__all__ = ["CONFIG", "configure_distributed", "device", "is_autocast_available"]


def _normalize_identifier(value: str) -> str:
    """Convert arbitrary text into a filesystem-safe identifier fragment."""
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    value = value.strip("._-")
    return value or "unknown"


class Configurations:
    """Singleton container for runtime, device, and distributed settings."""

    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
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
            identifier = self._get_distributed_identifier()
            torchinductor_root = os.getenv("TORCHINDUCTOR_CACHE_DIR", f"/tmp/cache/torchinductor/{torch.__version__}")
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(torchinductor_root, identifier)
            if (triton_root := os.getenv("TRITON_CACHE_DIR")) is None:
                os.environ["TRITON_CACHE_DIR"] = os.path.join(os.environ["TORCHINDUCTOR_CACHE_DIR"], "triton")
            else:
                os.environ["TRITON_CACHE_DIR"] = os.path.join(triton_root, identifier)
            try:
                import warp as wp

                warp_root = os.getenv("WARP_CACHE_PATH", f"/tmp/cache/warp/{wp.__version__}")
                os.environ["WARP_CACHE_PATH"] = os.path.join(warp_root, identifier)
            except ImportError:
                pass
        else:
            self._distributed = False
            self._rank = 0
            self._local_rank = 0
            self._world_size = 1
            self._local_world_size = 1

    @property
    def cuda(self) -> bool:
        """Whether CUDA is available in the current runtime."""
        return self._cuda

    @property
    def seed(self) -> int | None:
        """Configured global random seed, if one has been assigned."""
        return self._seed

    @seed.setter
    def seed(self, value: int | None):
        self._seed = value

    @property
    def device(self) -> torch.device:
        """Primary device used by the current process."""
        return self._device

    @device.setter
    def device(self, value: str | torch.device):
        self._device = torch.device(value)

    def set_device(self, value: str | torch.device):
        """Update the configured default device for the current process."""
        self.device = value

    @property
    def distributed(self) -> bool:
        """Whether the current process was launched in distributed mode."""
        return self._distributed

    @property
    def rank(self) -> int:
        """Global rank of the current process."""
        return self._rank

    @property
    def local_rank(self) -> int:
        """Rank of the current process within its local node."""
        return self._local_rank

    @property
    def world_size(self) -> int:
        """Total number of processes participating in the job."""
        return self._world_size

    @property
    def local_world_size(self) -> int:
        """Number of processes participating on the current node."""
        return self._local_world_size

    def _get_distributed_identifier(self) -> str:
        """Build a process-specific cache directory suffix for distributed runs."""
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
    """Return whether autocast is available on the configured default device."""
    return CONFIG.cuda and torch.amp.autocast_mode.is_autocast_available(CONFIG.device.type)


def configure_distributed(
    backend: str | None = None,
    **kwargs,
) -> bool:
    """Initialize the distributed process group when distributed training is enabled."""
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
    """Destroy the distributed process group during interpreter shutdown."""
    if CONFIG.distributed and GroupMember.WORLD is not None:
        if CONFIG.rank == 0:
            print("\033[1;32mCleaning distributed training resources.\033[0m")
        torch.distributed.destroy_process_group()
