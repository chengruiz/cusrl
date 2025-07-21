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
    Factory = DenormalizationFactory

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return input * self.std + self.mean
