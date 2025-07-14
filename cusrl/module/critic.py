from dataclasses import dataclass

import torch
from torch import nn

from cusrl.module.module import Module, ModuleFactory, ModuleFactoryLike
from cusrl.utils import RunningMeanStd, make_distributed
from cusrl.utils.typing import Memory, Slice

__all__ = ["Value"]


@dataclass(slots=True)
class ValueFactory(ModuleFactory["Value"]):
    backbone_factory: ModuleFactoryLike
    latent_dim: int | None = None

    def __call__(self, input_dim: int | None, output_dim: int = 1) -> "Value":
        return Value(self.backbone_factory(input_dim, self.latent_dim), output_dim)


class Value(Module):
    Factory = ValueFactory
    backbone: Module

    def __init__(self, backbone: Module, value_dim: int = 1):
        super().__init__(
            input_dim=backbone.input_dim,
            output_dim=value_dim,
            is_recurrent=backbone.is_recurrent,
        )
        self.backbone = backbone.rnn_compatible()
        self.value_head = nn.Linear(backbone.output_dim, value_dim)
        self.value_rms: RunningMeanStd | None = None

    def to_distributed(self):
        if not self.is_distributed:
            self.is_distributed = True
            self.backbone = self.backbone.to_distributed()
            self.value_head = make_distributed(self.value_head)
        return self

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        memory: Memory = None,
        done: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        value, _ = self(state, action, memory=memory, done=done, **kwargs)
        return value

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        memory: Memory = None,
        done: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Memory]:
        if action is not None:
            raise ValueError("State value function V(s) should not accept actions as input.")
        latent, memory = self.backbone(state, memory=memory, done=done, **kwargs)
        self.intermediate_repr["backbone.output"] = latent
        self.intermediate_repr["backbone.intermediate_repr"] = self.backbone.intermediate_repr
        return self.value_head(latent), memory

    def step_memory(self, state, memory=None, **kwargs):
        return self.backbone.step_memory(state, memory, **kwargs)

    def reset_memory(self, memory: Memory, done: Slice | torch.Tensor | None = None):
        return self.backbone.reset_memory(memory, done=done)
