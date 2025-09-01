import torch
from torch import nn

try:
    import flash_attn
    from flash_attn import flash_attn_func, flash_attn_kvpacked_func
except ImportError:
    flash_attn = flash_attn_func = flash_attn_kvpacked_func = None


__all__ = [
    "FlashAttention",
    "MultiheadAttention",
    "MultiheadCrossAttention",
]


class FlashAttention(nn.Module):
    @classmethod
    def is_available(cls) -> bool:
        return flash_attn is not None and torch.cuda.is_available()

    def __init__(self):
        if flash_attn is None:
            raise ImportError("FlashAttention is not installed. See https://github.com/Dao-AILab/flash-attention.")
        if not torch.cuda.is_available():
            raise RuntimeError("FlashAttention requires a CUDA-capable device.")
        super().__init__()


class MultiheadAttention(nn.Module):
    """Implements a multi-head attention layer.

    This module provides a multi-head attention mechanism that uses
    FlashAttention when available for optimal performance, and falls back to
    PyTorch's scaled_dot_product_attention otherwise.

    Args:
        embed_dim (int):
            Total dimension of the model.
        num_heads (int):
            The number of parallel attention heads.
        dropout (float, optional):
            Dropout probability on attention weights. Defaults to 0.0.
        bias (bool, optional):
            If ``True``, add bias to the input and output projection layers.
            Defaults to True.
        k_dim (int | None, optional):
            The dimension of the key tensor. If ``None``, defaults to
            `embed_dim`. Defaults to None.
        v_dim (int | None, optional):
            The dimension of the value tensor. If ``None``, defaults to
            `embed_dim`. Defaults to None.
        batch_first (bool, optional):
            If ``True``, then the input and output tensors are provided as
            (batch, sequence, channel). Defaults to True.
        dtype (torch.dtype, optional):
            The data type for the FlashAttention computation. Only
            `torch.float16` and `torch.bfloat16` are supported for
            FlashAttention. Defaults to `torch.float16`.

    Raises:
        ValueError: If `embed_dim` is not divisible by `num_heads`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        k_dim: int | None = None,
        v_dim: int | None = None,
        batch_first: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k_dim = embed_dim if k_dim is None else int(k_dim)
        self.v_dim = embed_dim if v_dim is None else int(v_dim)

        self.dropout = dropout
        self.batch_first = batch_first
        self.dtype = dtype
        self._use_flash_attn = FlashAttention.is_available() and dtype in (torch.float16, torch.bfloat16)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.k_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.v_dim, embed_dim, bias=bias)
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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Transpose inputs to (B, L, E)
        if not self.batch_first:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        # Projections
        q = self.q_proj(q).unflatten(-1, (self.num_heads, self.head_dim))
        k = self.k_proj(k).unflatten(-1, (self.num_heads, self.head_dim))
        v = self.v_proj(v).unflatten(-1, (self.num_heads, self.head_dim))

        if self._use_flash_attn and q.device.type != "cpu":
            assert flash_attn_func is not None
            attn_out = flash_attn_func(
                q.to(self.dtype),
                k.to(self.dtype),
                v.to(self.dtype),
                dropout_p=self.dropout if self.training else 0.0,
                causal=is_causal,
                return_attn_probs=False,
            )
            assert isinstance(attn_out, torch.Tensor)
        else:
            # Fallback to PyTorch's scaled_dot_product_attention
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(-2, -3),
                k.transpose(-2, -3),
                v.transpose(-2, -3),
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            ).transpose(-2, -3)

        attn_out = self.out_proj(attn_out.flatten(-2, -1).type_as(q))

        # Project back to (L, B, E) if batch_first=False
        if not self.batch_first:
            attn_out = attn_out.transpose(0, 1)

        return attn_out, None


class MultiheadCrossAttention(nn.Module):
    """Multi-head cross-attention module.

    This module computes cross-attention between a query tensor `q` and a
    key/value tensor `kv`. It uses a single linear projection for both key and
    value from the `kv` input, which are then processed in a packed format by
    the `flash_attn_kvpacked_func` for efficiency if FlashAttention is enabled.

    Args:
        embed_dim (int):
            Total dimension of the model.
        num_heads (int):
            The number of parallel attention heads.
        dropout (float, optional):
            Dropout probability on attention weights. Defaults to 0.0.
        bias (bool, optional):
            If ``True``, add bias to the input and output projection layers.
            Defaults to True.
        kv_dim (int | None, optional):
            If ``None``, defaults to `embed_dim`. Defaults to None.
        batch_first (bool, optional):
            If ``True``, then the input and output tensors are provided as
            (batch, sequence, channel). Defaults to True.
        dtype (torch.dtype, optional):
            The data type for the FlashAttention computation. Only
            `torch.float16` and `torch.bfloat16` are supported for
            FlashAttention. Defaults to `torch.float16`.

    Raises:
        ValueError: If `embed_dim` is not divisible by `num_heads`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        kv_dim: int | None = None,
        batch_first: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Unified key/value input dimension
        self.kv_dim = embed_dim if kv_dim is None else int(kv_dim)

        self.dropout = dropout
        self.batch_first = batch_first
        self.dtype = dtype
        self._use_flash_attn = FlashAttention.is_available() and dtype in (torch.float16, torch.bfloat16)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.kv_proj = nn.Linear(self.kv_dim, 2 * embed_dim, bias=bias)
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

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Transpose inputs to (B, L, E)
        if not self.batch_first:
            q = q.transpose(0, 1)
            kv = kv.transpose(0, 1)

        # Projections
        q = self.q_proj(q).unflatten(-1, (self.num_heads, self.head_dim))
        kv = self.kv_proj(kv).unflatten(-1, (2, self.num_heads, self.head_dim))

        if self._use_flash_attn and q.device.type != "cpu":
            # Compute cross-attention via FlashAttention with KV packed
            assert flash_attn_kvpacked_func is not None
            attn_out = flash_attn_kvpacked_func(
                q.to(self.dtype),
                kv.to(self.dtype),
                dropout_p=self.dropout if self.training else 0.0,
                causal=False,
                return_attn_probs=False,
            )
            assert isinstance(attn_out, torch.Tensor)
        else:
            # Fallback to PyTorch's scaled_dot_product_attention
            k, v = kv.unbind(dim=-3)

            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(-2, -3),
                k.transpose(-2, -3),
                v.transpose(-2, -3),
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            ).transpose(-2, -3)

        attn_out = self.out_proj(attn_out.flatten(-2, -1).type_as(q))

        # Project back to (L, B, E) if batch_first=False
        if not self.batch_first:
            attn_out = attn_out.transpose(0, 1)

        return attn_out, None
