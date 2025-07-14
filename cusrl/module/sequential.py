from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from cusrl.module.module import Module, ModuleFactory
from cusrl.utils.typing import Memory, Slice

__all__ = ["Sequential"]


@dataclass(slots=True)
class SequentialFactory(ModuleFactory["Sequential"]):
    factories: Iterable[ModuleFactory]
    hidden_dims: Iterable[int | None]

    def __call__(self, input_dim: int | None, output_dim: int | None):
        output_dims = list(self.hidden_dims) + [output_dim]
        layers = []
        for factory, output_dim in zip(self.factories, output_dims, strict=True):
            layer = factory(input_dim, output_dim)
            layers.append(layer)
            input_dim = layer.output_dim
        return Sequential(layers)


class Sequential(Module):
    Factory = SequentialFactory

    def __init__(self, modules: Iterable[Module]):
        self.layers = list(modules)
        super().__init__(
            input_dim=self.layers[0].input_dim,
            output_dim=self.layers[-1].output_dim,
            is_recurrent=any(layer.is_recurrent for layer in self.layers),
        )
        self.layers = nn.ModuleList(self.layers)

    def forward(self, input, **kwargs):
        last_memory = kwargs.pop("memory", None)
        if last_memory is not None:
            last_memory = iter(last_memory)
        memory = []
        for i, layer in enumerate(self.layers):
            if layer.is_recurrent:
                layer_memory = None if last_memory is None else next(last_memory)
                input, layer_memory = layer(input, memory=layer_memory, **kwargs)
                memory.append(layer_memory)
            else:
                input = layer(input, **kwargs)
            prefix = f"{i}/{type(layer).__name__}"
            self.intermediate_repr[f"{prefix}.output"] = input
            self.intermediate_repr[f"{prefix}.intermediate_repr"] = layer.intermediate_repr
        if memory:
            return input, tuple(memory)
        return input

    def clear_intermediate_repr(self):
        super().clear_intermediate_repr()
        for layer in self.layers:
            layer.clear_intermediate_repr()

    def reset_memory(self, memory: Memory, done: Slice | torch.Tensor | None = None):
        if not self.is_recurrent or memory is None:
            return
        memory = iter(memory)
        for layer in self.layers:
            if layer.is_recurrent:
                layer.reset_memory(next(memory), done)
