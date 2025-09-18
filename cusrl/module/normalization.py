from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from cusrl.module.module import Module, ModuleFactory

__all__ = ["Normalization", "Denormalization"]


@dataclass(slots=True)
class NormalizationFactory(ModuleFactory["Normalization"]):
    mean: Sequence[float] | np.ndarray | Tensor
    std: Sequence[float] | np.ndarray | Tensor

    def __call__(self, input_dim: int | None, output_dim: int | None):
        module = Normalization(torch.as_tensor(self.mean), torch.as_tensor(self.std))
        if input_dim is not None and module.input_dim != input_dim:
            raise ValueError(f"Input dimension mismatch: {module.input_dim} != {input_dim}.")
        if output_dim is not None and module.output_dim != output_dim:
            raise ValueError(f"Output dimension mismatch: {module.output_dim} != {output_dim}.")
        return module


class Normalization(Module):
    r"""Normalizes input tensors using a given mean and standard deviation.

    This module performs element-wise normalization on an input tensor with:
    .. math::
        \text{output} = (\text{input} - \text{mean}) / \text{std}.

    The :math:`\text{mean}` and :math:`\text{std}` tensors are provided on
    initialization and are stored as non-trainable parameters.

    Args:
        mean (Tensor):
            The mean tensor to be subtracted from the input.
        std (Tensor):
            The standard deviation tensor to divide the input by.
    """

    Factory = NormalizationFactory

    def __init__(self, mean: Tensor, std: Tensor):
        super().__init__(mean.size(0), mean.size(0))
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return (input - self.mean) / self.std


@dataclass(slots=True)
class DenormalizationFactory(ModuleFactory["Denormalization"]):
    mean: Sequence[float] | np.ndarray | Tensor
    std: Sequence[float] | np.ndarray | Tensor

    def __call__(self, input_dim: int | None, output_dim: int | None):
        module = Denormalization(torch.as_tensor(self.mean), torch.as_tensor(self.std))
        if input_dim is not None and module.input_dim != input_dim:
            raise ValueError(f"Input dimension mismatch: {module.input_dim} != {input_dim}.")
        if output_dim is not None and module.output_dim != output_dim:
            raise ValueError(f"Output dimension mismatch: {module.output_dim} != {output_dim}.")
        return module


class Denormalization(Normalization):
    r"""Denormalizes a tensor using a given mean and standard deviation.

    This module reverses the normalization process by scaling the input tensor
    back to its original data distribution with
    .. math::
        \text{output} = \text{input} * \text{std} + \text{mean}.

    It is the inverse operation of the :class:`Normalization` module.

    Args:
        mean (Tensor):
            The mean tensor to be added to the input after scaling.
        std (Tensor):
            The standard deviation tensor to scale the input by.
    """

    Factory = DenormalizationFactory

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return input * self.std + self.mean
