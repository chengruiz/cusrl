from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import overload

import torch
from torch import nn

from cusrl.nn.module.module import Module, ModuleFactory, ModuleFactoryLike
from cusrl.utils.dict_utils import prefix_dict_keys
from cusrl.utils.typing import Memory, Slice

__all__ = ["Sequential", "SequentialFactory"]


@dataclass(slots=True)
class SequentialFactory(ModuleFactory["Sequential"]):
    factories: Sequence[ModuleFactoryLike]
    hidden_dims: Sequence[int | None]

    def __call__(self, input_dim: int | None = None, output_dim: int | None = None):
        if len(self.hidden_dims) != len(self.factories) - 1:
            raise ValueError("'hidden_dims' must have one fewer element than 'factories'")

        layers: list[Module] = []
        for factory, hidden_dim in zip(self.factories, [*self.hidden_dims, output_dim], strict=True):
            layer = factory(input_dim, hidden_dim)
            input_dim = layer.output_dim
            layers.append(layer)
        return Sequential(layers)


class Sequential(Module):
    Factory = SequentialFactory

    @overload
    def __init__(self, modules: Module, *args: Module): ...
    @overload
    def __init__(self, modules: Iterable[Module]): ...

    def __init__(self, modules, *args):
        if args:
            assert isinstance(modules, Module)
            layers: list[Module] = [modules, *args]
        else:
            layers = [modules] if isinstance(modules, Module) else list(modules)
        super().__init__(
            input_dim=layers[0].input_dim,
            output_dim=layers[-1].output_dim,
            is_recurrent=any(layer.is_recurrent for layer in layers),
        )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        input: torch.Tensor,
        memory: Memory | None = None,
        *,
        done: torch.Tensor | None = None,
        sequential: bool = True,
        **kwargs,
    ):
        final_memory = {}
        for i, layer in enumerate(self.layers):
            layer_memory = None if memory is None else memory.get(str(i))
            input, layer_memory = layer.forward_with_memory(
                input,
                memory=layer_memory,
                done=done,
                sequential=sequential,
                **kwargs,
            )
            if layer_memory is not None:
                final_memory[str(i)] = layer_memory

            prefix = f"{i}/{type(layer).__name__}"
            self.intermediate_repr[f"{prefix}.output"] = input
            self.intermediate_repr.update(prefix_dict_keys(layer.intermediate_repr, f"{prefix}."))
        return (input, final_memory) if self.is_recurrent else input

    def clear_intermediate_repr(self):
        super().clear_intermediate_repr()
        for layer in self.layers:
            layer.clear_intermediate_repr()

    def reset_memory(self, memory: Memory, done: Slice | torch.Tensor | None = None):
        if not self.is_recurrent or memory is None:
            return
        for i, layer in enumerate(self.layers):
            if layer.is_recurrent:
                layer.reset_memory(memory.get(str(i)), done)
