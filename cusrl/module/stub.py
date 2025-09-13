from dataclasses import dataclass

import torch

from cusrl.module.module import Module, ModuleFactory

__all__ = ["StubModule"]


@dataclass(slots=True)
class StubModuleFactory(ModuleFactory["StubModule"]):
    def __call__(self, input_dim: int, output_dim: int | None):
        return StubModule(input_dim=input_dim, output_dim=output_dim)


class StubModule(Module):
    """A stub module serves as a placeholder in a model architecture."""

    Factory = StubModuleFactory

    def __init__(self, input_dim: int, output_dim: int | None):
        super().__init__(input_dim, output_dim or 1)

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return input.new_zeros((*input.shape[:-1], self.output_dim))
