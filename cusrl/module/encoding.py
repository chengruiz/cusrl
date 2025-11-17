import torch
from torch import Tensor, nn

from cusrl.utils.config import CONFIG

try:
    from flash_attn.layers.rotary import apply_rotary_emb as apply_rotary_emb_flash
    from flash_attn.layers.rotary import apply_rotary_emb_qkv_ as apply_rotary_emb_qkv_flash_
except ImportError:
    apply_rotary_emb_flash = apply_rotary_emb_qkv_flash_ = None


__all__ = [
    "LearnablePositionalEncoding2D",
    "RotaryEmbedding",
    "SinusoidalPositionalEncoding2D",
]


def sinusoidal_positional_encoding_2d(
    height: int,
    width: int,
    num_channels: int,
    *,
    base: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create 2D sinusoidal positional encodings (H, W, C)."""

    if num_channels % 4 != 0:
        raise ValueError(f"num_channels must be divisible by 4 for 2D sinusoidal encoding, got {num_channels}")

    half_channels = num_channels // 2
    h_positions = torch.arange(height, device=device, dtype=dtype).unsqueeze(1)  # (H, 1)
    w_positions = torch.arange(width, device=device, dtype=dtype).unsqueeze(1)  # (W, 1)

    div_y = torch.pow(
        torch.tensor(base, device=device, dtype=dtype),
        torch.arange(0, half_channels, 2, device=device, dtype=dtype) / half_channels,
    )  # (C / 4,)
    div_x = torch.pow(
        torch.tensor(base, device=device, dtype=dtype),
        torch.arange(0, half_channels, 2, device=device, dtype=dtype) / half_channels,
    )  # (C / 4,)

    angles_y = h_positions / div_y  # (H, C / 4)
    y_embed = torch.zeros(height, half_channels, device=device, dtype=dtype)
    y_embed[:, 0::2] = torch.sin(angles_y)
    y_embed[:, 1::2] = torch.cos(angles_y)

    angles_x = w_positions / div_x  # (W, C / 4)
    x_embed = torch.zeros(width, half_channels, device=device, dtype=dtype)
    x_embed[:, 0::2] = torch.sin(angles_x)
    x_embed[:, 1::2] = torch.cos(angles_x)

    pe = torch.zeros(height, width, num_channels, device=device, dtype=dtype)
    pe[:, :, :half_channels] = y_embed.unsqueeze(1).expand(height, width, half_channels)
    pe[:, :, half_channels:] = x_embed.unsqueeze(0).expand(height, width, half_channels)
    return pe


class SinusoidalPositionalEncoding2D(nn.Module):
    """Adds 2D sinusoidal positional encodings to an input tensor.

    The shape of the input tensor should be :math:`(..., C, H, W)`, where
    :math:`C` is the number of channels, :math:`H` is the height, and :math:`W`
    is the width. The number of channels :math:`C` must be divisible by 4.

    Args:
        num_channels (int):
            The number of channels of the input tensor.
        height (int):
            The height of the input tensor's spatial dimensions.
        width (int):
            The width of the input tensor's spatial dimensions.
        base (float, optional):
            The base value for the sinusoidal formula. Defaults to ``10000.0``.
        requires_grad (bool, optional):
            If ``True``, make the positional encodings learnable. Defaults to
            ``False``.
    """

    def __init__(self, num_channels: int, height: int, width: int, base: float = 10000.0, requires_grad: bool = False):
        super().__init__()
        pe = sinusoidal_positional_encoding_2d(height, width, num_channels, base=base).permute(2, 0, 1)
        self.pe = nn.Parameter(pe, requires_grad=requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe.type_as(x)


class LearnablePositionalEncoding2D(nn.Module):
    """Adds learnable 2D positional encodings to the input tensor.

    The shape of the input tensor should be :math:`(..., C, H, W)`, where
    :math:`C` is the number of channels, :math:`H` is the height, and :math:`W`
    is the width.

    Args:
        num_channels (int):
            The number of channels of the input tensor.
        height (int):
            The height of the input tensor's spatial dimensions.
        width (int):
            The width of the input tensor's spatial dimensions.
    """

    def __init__(self, num_channels: int, height: int, width: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(num_channels, height, width))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe.type_as(x)


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embedding to tensor.

    Args:
        x: tensor of shape (B, L, H, C / H).
        cos, sin: tensor of shape (L, C / H / 2).
    """
    cos = cos.repeat(1, 2).unsqueeze(-2)  # (L, 1, C / H)
    sin = sin.repeat(1, 2).unsqueeze(-2)  # (L, 1, C / H)
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    """Implements Rotary Positional Embedding (RoPE), described in:
    "RoFormer: Enhanced Transformer with Rotary Position Embedding",
    https://arxiv.org/abs/2104.09864.

    Args:
        head_dim (int):
            The per-head dimensionality of the feature embedding. Must be an
            even number.
        max_seq_len (int, optional):
            The maximum sequence length for which to pre-cache the sine and
            cosine values. Defaults to 2048.
        base (float, optional):
            The base value used for calculating the inverse frequency of the
            sinusoidal embeddings. Defaults to 10000.0.

    Raises:
        ValueError: If `num_channels` is not an even number.
    """

    inv_freq: Tensor
    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("'head_dim' must be even for RotaryEmbedding.")
        self.head_dim = head_dim
        self.max_seq_len = int(max_seq_len)
        self.base = float(base)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._build_cache(self.max_seq_len)

        from cusrl.module.mha import FlashAttention

        self._flash = FlashAttention.is_available()

    def _build_cache(self, seq_len: int) -> None:
        positions = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        angles = positions.unsqueeze(1) @ self.inv_freq.unsqueeze(0)  # (L, C / 2)
        self.cos_cached = torch.cos(angles)
        self.sin_cached = torch.sin(angles)

    def _get_cos_sin(self, seq_len: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)
        if dtype is not None:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
        return cos, sin

    def forward(self, x: Tensor) -> Tensor:
        cos, sin = self._get_cos_sin(x.shape[-3], device=x.device, dtype=x.dtype)
        if self._flash and CONFIG.flash_attention_enabled and x.device.type != "cpu":
            assert apply_rotary_emb_flash is not None
            return apply_rotary_emb_flash(x, cos, sin)  # type: ignore
        return apply_rotary_emb(x, cos, sin)

    def apply_qkv(self, qkv: Tensor) -> Tensor:
        """Apply rotary embedding to query, key, and value tensors.

        Args:
            qkv: tensor of shape (B, L, 3, H, C / H).
        """
        if self._flash and CONFIG.flash_attention_enabled and qkv.device.type != "cpu":
            assert apply_rotary_emb_qkv_flash_ is not None
            cos, sin = self._get_cos_sin(qkv.shape[-4], device=qkv.device, dtype=qkv.dtype)
            apply_rotary_emb_qkv_flash_(qkv, cos, sin)
            return qkv
        q, k, v = qkv.unbind(dim=-3)
        return torch.stack([self(q), self(k), v], dim=-3)
