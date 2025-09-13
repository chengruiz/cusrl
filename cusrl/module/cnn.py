import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn

from cusrl.module.module import Module, ModuleFactory

__all__ = ["Cnn", "SeparableConv2d"]


@dataclass(slots=True)
class CnnFactory(ModuleFactory["Cnn"]):
    layer_factories: Iterable[Callable[[], nn.Module]]
    input_shape: tuple[int, int] | tuple[int, int, int]
    input_flattened: bool = True
    flatten_output: bool = True

    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        module = Cnn(
            [factory() for factory in self.layer_factories],
            input_shape=self.input_shape,
            input_flattened=self.input_flattened,
            flatten_output=self.flatten_output,
            output_dim=output_dim,
        )
        if input_dim is not None and module.input_dim != input_dim:
            raise ValueError(f"Input dimension mismatch ({module.input_dim} != {input_dim}).")
        return module


class Cnn(Module):
    """A generic Convolutional Neural Network (CNN) module.

    This module wraps a sequence of PyTorch layers to form a CNN. It handles
    input reshaping based on `input_shape` and `input_flattened`. It can also
    flatten the output of the convolutional layers and optionally add a final
    linear layer to project the features to a specific dimension.

    The module supports inputs with multiple batch dimensions.

    Args:
        layers (Iterable[nn.Module | Module]):
            A sequence of modules (e.g., `nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`)
            that constitute the convolutional part of the network.
        input_shape (tuple[int, int] | tuple[int, int, int]):
            The shape of the input data, either `(height, width)` or `(channel,
            height, width)`.
        input_flattened (bool, optional):
            If `True`, the input tensor is expected to be a flat vector and will
            be unflattened to `input_shape` before being passed through the
            layers. Defaults to True.
        flatten_output (bool, optional):
            If `True`, the output of the final convolutional layer is flattened
            into a 1D tensor. Defaults to True.
        output_dim (int | None, optional):
            If specified, a `nn.Linear` layer is appended to map the flattened
            features to this output dimension. `flatten_output` is implicitly
            True if this is set. Defaults to None.
    """

    Factory = CnnFactory

    def __init__(
        self,
        layers: Iterable[nn.Module | Module],
        input_shape: tuple[int, int] | tuple[int, int, int],
        input_flattened: bool = True,
        flatten_output: bool = True,
        output_dim: int | None = None,
    ):
        layers = nn.Sequential(*layers)
        if len(input_shape) not in (2, 3):
            raise ValueError("'input_shape' should be 2- or 3-dimensional.")
        if len(input_shape) == 2:
            # add channel dimension if missing
            input_shape = (1, *input_shape)

        super().__init__(math.prod(input_shape), layers(torch.zeros(input_shape)).numel())
        self.input_shape = input_shape

        # convolution layers
        self.layers: nn.Sequential = layers
        self.input_flattened = input_flattened
        if output_dim is not None:
            self.layers.append(nn.Flatten(-3))  # flatten [channel, y, x]
            self.layers.append(nn.Linear(self.output_dim, output_dim))
            self.output_dim = output_dim
        elif flatten_output:
            self.layers.append(nn.Flatten(-3))

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.input_flattened:
            input = input.unflatten(-1, self.input_shape)

        # Flatten batch dimensions
        batch_dims = input.shape[:-3]
        if batch_dims:
            input = input.flatten(0, -4)
        output = self.layers(input)
        if batch_dims:
            output = output.unflatten(0, batch_dims)
        return output


class SeparableConv2d(nn.Module):
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
        device=None,
        dtype=None,
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

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
