from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from cusrl.module.module import Module, ModuleFactory

__all__ = ["Mlp"]


@dataclass(slots=True)
class MlpFactory(ModuleFactory["Mlp"]):
    hidden_dims: Iterable[int]
    activation_fn: str | type[nn.Module] = nn.ReLU
    ends_with_activation: bool = False
    dropout: float = 0.0

    def __call__(self, input_dim: int, output_dim: int | None):
        return Mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=self.hidden_dims,
            activation_fn=self._resolve_activation_fn(self.activation_fn),
            ends_with_activation=self.ends_with_activation,
            dropout=self.dropout,
        )


class Mlp(Module):
    """A multi-layer perceptron (MLP) module.

    This class builds a sequential neural network model consisting of linear
    layers, activation functions, and optional dropout layers. The architecture
    can be flexibly defined by specifying the dimensions of the input, hidden,
    and output layers.

    Args:
        input_dim (int):
            The dimension of the input features.
        hidden_dims (Iterable[int]):
            An iterable of integers specifying the size of each hidden layer.
        output_dim (int | None, optional):
            The dimension of the output. If `None`, the last element of
            `hidden_dims` is used as the output dimension, and the preceding
            elements define the hidden layers. Defaults to `None`.
        activation_fn (type[nn.Module], optional):
            The activation function class to be used after each hidden layer.
            Defaults to `nn.ReLU`.
        ends_with_activation (bool, optional):
            If `True`, an activation function is applied to the final output
            layer. Defaults to `False`.
        dropout (float, optional):
            The dropout rate to be applied after each activation function in the
            hidden layers. A value of 0.0 means no dropout. Defaults to `0.0`.
    """

    Factory = MlpFactory

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        ends_with_activation: bool = False,
        dropout: float = 0.0,
    ):
        hidden_dims = list(hidden_dims)
        if output_dim is None:
            if not hidden_dims:
                raise ValueError("'hidden_dims' should not be empty if output_dim is None.")
            output_dim = hidden_dims[-1]
            hidden_dims = hidden_dims[:-1]
        super().__init__(input_dim, output_dim)

        self.layers = nn.Sequential()
        hidden_dims.insert(0, input_dim)
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(activation_fn())
            if dropout > 0.0:
                self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        if ends_with_activation:
            self.layers.append(activation_fn())

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.layers(input)
