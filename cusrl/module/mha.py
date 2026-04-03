from typing import Literal

import torch
from torch import Tensor, nn

try:
    import flash_attn
except ImportError:
    flash_attn = None

from cusrl.module.encoding import RotaryEmbedding

__all__ = [
    "FlashAttention",
    "MultiheadAttention",
    "MultiheadCrossAttention",
    "MultiheadSelfAttention",
]


def make_norm(norm: Literal["rms", "layer"] | None, head_dim: int) -> nn.Module:
    if norm is None:
        return nn.Identity()
    if norm == "rms":
        return nn.RMSNorm(head_dim, eps=1e-6)
    if norm == "layer":
        return nn.LayerNorm(head_dim, eps=1e-6)
    raise ValueError(f"Unsupported normalization type: {norm!r}")


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


class MultiheadAttention(nn.Module):
    """Implements a multi-head attention layer.

    This module provides a multi-head attention mechanism based on PyTorch's
    scaled dot product attention.

    Args:
        embed_dim (int):
            Total dimension of the model.
        num_heads (int):
            The number of parallel attention heads.
        dropout (float, optional):
            Dropout probability on attention weights. Defaults to ``0.0``.
        bias (bool, optional):
            If ``True``, add bias to the input and output projection layers.
            Defaults to ``True``.
        k_dim (int | None, optional):
            The dimension of the key tensor. If ``None``, defaults to
            ``embed_dim``. Defaults to None.
        v_dim (int | None, optional):
            The dimension of the value tensor. If ``None``, defaults to
            ``embed_dim``. Defaults to None.
        batch_first (bool, optional):
            If ``True``, then the input and output tensors are provided as
            (batch, sequence, channel). Defaults to ``True``.
        dtype (torch.dtype, optional):
            The data type used for the attention computation. Defaults to
            ``torch.float32``.

    Raises:
        ValueError: If ``embed_dim`` is not divisible by ``num_heads``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qk_norm: Literal["rms", "layer"] | None = None,
        bias: bool = True,
        k_dim: int | None = None,
        v_dim: int | None = None,
        batch_first: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"'embed_dim' ({embed_dim}) must be divisible by 'num_heads' ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k_dim = embed_dim if k_dim is None else int(k_dim)
        self.v_dim = embed_dim if v_dim is None else int(v_dim)

        self.dropout = dropout
        self.batch_first = batch_first
        self.dtype = dtype

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.k_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.v_dim, embed_dim, bias=bias)
        self.q_norm = make_norm(qk_norm, self.head_dim)
        self.k_norm = make_norm(qk_norm, self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.0)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False) -> Tensor:
        # Transpose inputs to (N, L, E)
        if not self.batch_first:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        # Projections
        q = self.q_proj(q).unflatten(-1, (self.num_heads, self.head_dim))
        k = self.k_proj(k).unflatten(-1, (self.num_heads, self.head_dim))
        v = self.v_proj(v).unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q.to(self.dtype).transpose(-2, -3),
            k.to(self.dtype).transpose(-2, -3),
            v.to(self.dtype).transpose(-2, -3),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        ).transpose(-2, -3)
        attn_out = self.out_proj(attn_out.flatten(-2, -1).type_as(q))

        # Project back to (L, N, E) if batch_first=False
        if not self.batch_first:
            attn_out = attn_out.transpose(0, 1)
        return attn_out


class MultiheadCrossAttention(nn.Module):
    """Multi-head cross-attention module.

    This module computes cross-attention between a query tensor ``q`` and a
    key/value tensor ``kv``. It uses a single linear projection for both key and
    value from the ``kv`` input, and computes attention with PyTorch SDPA.

    Args:
        embed_dim (int):
            Total dimension of the model.
        num_heads (int):
            The number of parallel attention heads.
        dropout (float, optional):
            Dropout probability on attention weights. Defaults to ``0.0``.
        bias (bool, optional):
            If ``True``, add bias to the input and output projection layers.
            Defaults to ``True``.
        kv_dim (int | None, optional):
            If ``None``, defaults to ``embed_dim``. Defaults to None.
        batch_first (bool, optional):
            If ``True``, then the input and output tensors are provided as
            (batch, sequence, channel). Defaults to ``True``.
        dtype (torch.dtype, optional):
            The data type used for the attention computation. Defaults to
            ``torch.float32``.

    Raises:
        ValueError: If ``embed_dim`` is not divisible by ``num_heads``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qk_norm: Literal["rms", "layer"] | None = None,
        bias: bool = True,
        kv_dim: int | None = None,
        batch_first: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"'embed_dim' ({embed_dim}) must be divisible by 'num_heads' ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Unified key/value input dimension
        self.kv_dim = embed_dim if kv_dim is None else int(kv_dim)

        self.dropout = dropout
        self.batch_first = batch_first
        self.dtype = dtype

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_proj = nn.Linear(self.kv_dim, 2 * embed_dim, bias=bias)
        self.q_norm = make_norm(qk_norm, self.head_dim)
        self.k_norm = make_norm(qk_norm, self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
        if self.kv_proj.bias is not None:
            nn.init.constant_(self.kv_proj.bias, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        # Transpose inputs to (N, L, E)
        if not self.batch_first:
            q = q.transpose(0, 1)
            kv = kv.transpose(0, 1)

        # Projections
        q = self.q_proj(q).unflatten(-1, (self.num_heads, self.head_dim))
        k, v = self.kv_proj(kv).unflatten(-1, (2, self.num_heads, self.head_dim)).unbind(dim=-3)
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q.to(self.dtype).transpose(-2, -3),
            k.to(self.dtype).transpose(-2, -3),
            v.to(self.dtype).transpose(-2, -3),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        ).transpose(-2, -3)
        attn_out = self.out_proj(attn_out.flatten(-2, -1).type_as(q))
        # Project back to (L, N, E) if batch_first=False
        if not self.batch_first:
            attn_out = attn_out.transpose(0, 1)
        return attn_out


class MultiheadSelfAttention(nn.Module):
    """Multi-head self-attention module.

    This module implements a multi-head self-attention mechanism. It uses a
    single linear projection for query, key and value from input, and computes
    attention with PyTorch SDPA.

    Args:
        embed_dim (int):
            Total dimension of the model.
        num_heads (int):
            The number of parallel attention heads.
        rope_base (float | None, optional):
            If provided, enables Rotary Positional Embedding with the given
            base frequency. Defaults to ``None``.
        dropout (float, optional):
            Dropout probability on attention weights. Defaults to ``0.0``.
        bias (bool, optional):
            If ``True``, add bias to the input and output projection layers.
            Defaults to ``True``.
        batch_first (bool, optional):
            If ``True``, then the input and output tensors are provided as
            (batch, sequence, channel). Defaults to ``True``.
        dtype (torch.dtype, optional):
            The data type used for the attention computation. Defaults to
            ``torch.float32``.

    Raises:
        ValueError: If ``embed_dim`` is not divisible by ``num_heads``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rope_base: float | None = None,
        dropout: float = 0.0,
        qk_norm: Literal["rms", "layer"] | None = None,
        bias: bool = True,
        batch_first: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"'embed_dim' ({embed_dim}) must be divisible by 'num_heads' ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rope_base = rope_base

        self.dropout = dropout
        self.batch_first = batch_first
        self.dtype = dtype

        self.rope = RotaryEmbedding(self.head_dim, base=rope_base) if rope_base is not None else None
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.q_norm = make_norm(qk_norm, self.head_dim)
        self.k_norm = make_norm(qk_norm, self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.qkv_proj.bias is not None:
            nn.init.constant_(self.qkv_proj.bias, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, input: Tensor, is_causal: bool = False) -> Tensor:
        # Transpose inputs to (N, L, E)
        if not self.batch_first:
            input = input.transpose(0, 1)

        # Projections
        qkv = self.qkv_proj(input).unflatten(-1, (3, self.num_heads, self.head_dim))
        if self.rope is not None:
            qkv = self.rope.apply_qkv(qkv)
        q, k, v = qkv.unbind(dim=-3)
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q.to(self.dtype).transpose(-2, -3),
            k.to(self.dtype).transpose(-2, -3),
            v.to(self.dtype).transpose(-2, -3),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        ).transpose(-2, -3)
        attn_out = self.out_proj(attn_out.flatten(-2, -1).type_as(q))

        # Project back to (L, N, E) if batch_first=False
        if not self.batch_first:
            attn_out = attn_out.transpose(0, 1)
        return attn_out
