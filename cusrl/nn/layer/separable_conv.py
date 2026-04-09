import torch
from torch import Tensor, nn

__all__ = ["SeparableConv2d"]


class SeparableConv2d(nn.Module):
    """Depthwise-separable 2D convolution.

    This layer applies a depthwise spatial convolution independently to each
    input channel, followed by a pointwise ``1 x 1`` convolution that mixes
    channels into the requested output dimension.

    Args:
        in_channels (int):
            Number of channels in the input tensor.
        out_channels (int):
            Number of channels produced by the pointwise projection.
        kernel_size (int | tuple[int, int]):
            Spatial kernel size for the depthwise convolution.
        stride (int | tuple[int, int], optional):
            Stride for the depthwise convolution. Defaults to ``1``.
        padding (str | int | tuple[int, int], optional):
            Padding applied by the depthwise convolution. Defaults to ``0``.
        dilation (int | tuple[int, int], optional):
            Dilation factor for the depthwise convolution. Defaults to ``1``.
        bias (bool, optional):
            Whether to include learnable bias terms in both convolutions.
            Defaults to ``True``.
        padding_mode (str, optional):
            Padding mode for the depthwise convolution. Defaults to ``"zeros"``.
        device (torch.device | str | None, optional):
            Device for parameter initialization. Defaults to ``None``.
        dtype (torch.dtype | None, optional):
            Dtype for parameter initialization. Defaults to ``None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: str | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the depthwise and pointwise convolutions in sequence.

        Args:
            x (Tensor):
                Input tensor of shape ``(N, C_in, H, W)``.

        Returns:
            Tensor:
                Output tensor of shape ``(N, C_out, H_out, W_out)``.
        """
        return self.pointwise(self.depthwise(x))
