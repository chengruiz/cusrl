import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn

from cusrl.module.module import Module, ModuleFactory

__all__ = ["Cnn"]


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
        if len(input_shape) == 1:
            raise ValueError("'input_shape' should be at least 2-dimensional.")
        if len(input_shape) == 2:
            # add channel dimension if missing
            input_shape = (1, *input_shape)

        super().__init__(math.prod(input_shape), layers(torch.zeros(input_shape)).numel())
        self.input_shape = input_shape

        # convolution layers
        self.layers = layers
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

        # enable multiple batch dimensions
        batch_dims = input.shape[:-3]
        if batch_dims:
            input = input.flatten(0, -4)
        output = self.layers(input)
        if batch_dims:
            output = output.unflatten(0, batch_dims)
        return output
