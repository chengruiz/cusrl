import torch
from torch import Tensor, nn

__all__ = ["LearnablePositionalEncoding2D", "SinusoidalPositionalEncoding2D"]


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
    """

    def __init__(self, num_channels: int, height: int, width: int, base: float = 10000.0):
        super().__init__()
        pe = sinusoidal_positional_encoding_2d(height, width, num_channels, base=base).permute(2, 0, 1)
        self.pe: Tensor
        self.register_buffer("pe", pe, persistent=False)

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
