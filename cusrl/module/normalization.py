from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from cusrl.module.module import Module, ModuleFactory

__all__ = ["Normalization", "Denormalization"]


@dataclass(slots=True)
class NormalizationFactory(ModuleFactory["Normalization"]):
    mean: Sequence[float] | np.ndarray | torch.Tensor
    std: Sequence[float] | np.ndarray | torch.Tensor

    def __call__(self, input_dim: int | None, output_dim: int | None):
        module = Normalization(torch.as_tensor(self.mean), torch.as_tensor(self.std))
        if input_dim is not None and module.input_dim != input_dim:
            raise ValueError(f"Input dimension mismatch: {module.input_dim} != {input_dim}.")
        if output_dim is not None and module.output_dim != output_dim:
            raise ValueError(f"Output dimension mismatch: {module.output_dim} != {output_dim}.")
        return module


class Normalization(Module):
    """Normalizes input tensors using a given mean and standard deviation.

    This module performs element-wise normalization on an input tensor using the
    formula: `output = (input - mean) / std`. The `mean` and `std` tensors are
    provided during initialization and are stored as non-trainable parameters.

    Args:
        mean (torch.Tensor):
            The mean tensor to be subtracted from the input.
        std (torch.Tensor):
            The standard deviation tensor to divide the input by.
    """

    Factory = NormalizationFactory

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__(mean.size(0), mean.size(0))
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return (input - self.mean) / self.std


@dataclass(slots=True)
class DenormalizationFactory(ModuleFactory["Denormalization"]):
    mean: Sequence[float] | np.ndarray | torch.Tensor
    std: Sequence[float] | np.ndarray | torch.Tensor

    def __call__(self, input_dim: int | None, output_dim: int | None):
        module = Denormalization(torch.as_tensor(self.mean), torch.as_tensor(self.std))
        if input_dim is not None and module.input_dim != input_dim:
            raise ValueError(f"Input dimension mismatch: {module.input_dim} != {input_dim}.")
        if output_dim is not None and module.output_dim != output_dim:
            raise ValueError(f"Output dimension mismatch: {module.output_dim} != {output_dim}.")
        return module


class Denormalization(Normalization):
    """Denormalizes a tensor using a given mean and standard deviation.

    This module reverses the normalization process by scaling the input tensor
    back to its original data distribution. The transformation is defined by the
    formula: `output = input * std + mean`. It is the inverse operation of the
    `Normalization` module.

    Args:
        mean (torch.Tensor):
            The mean tensor to be added to the input after scaling.
        std (torch.Tensor):
            The standard deviation tensor to scale the input by.
    """

    Factory = DenormalizationFactory

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return input * self.std + self.mean
