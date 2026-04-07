import torch
from torch import nn

try:
    import flash_attn
except ImportError:
    flash_attn = None

__all__ = ["FlashAttention"]


class FlashAttention(nn.Module):
    SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}

    @classmethod
    def is_available(cls, dtype: torch.dtype | None = None) -> bool:
        return flash_attn is not None and torch.cuda.is_available() and (dtype is None or dtype in cls.SUPPORTED_DTYPES)

    def __init__(self):
        if flash_attn is None:
            raise ImportError("FlashAttention is not installed; see https://github.com/Dao-AILab/flash-attention")
        if not torch.cuda.is_available():
            raise RuntimeError("FlashAttention requires a CUDA-capable device")
        super().__init__()
