from dataclasses import dataclass

import torch

from cusrl.module.module import Module, ModuleFactory

__all__ = ["Identity", "StubModule"]


@dataclass(slots=True)
class StubModuleFactory(ModuleFactory["StubModule"]):
    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        assert input_dim is not None
        return StubModule(input_dim=input_dim, output_dim=output_dim)


class StubModule(Module):
    """A stub module serves as a placeholder in a model architecture."""

    Factory = StubModuleFactory

    def __init__(self, input_dim: int, output_dim: int | None):
        super().__init__(input_dim, output_dim or 1)

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return input.new_zeros((*input.shape[:-1], self.output_dim))


@dataclass(slots=True)
class IdentityFactory(ModuleFactory["Identity"]):
    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        assert input_dim is not None
        assert output_dim is None or output_dim == input_dim
        return Identity(input_dim=input_dim, output_dim=output_dim)


class Identity(Module):
    """An identity module that returns the input as is."""

    Factory = IdentityFactory

    def __init__(self, input_dim: int, output_dim: int | None):
        super().__init__(input_dim, output_dim or input_dim)
        assert self.input_dim == self.output_dim

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return input
