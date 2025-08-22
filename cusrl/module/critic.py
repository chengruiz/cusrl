from dataclasses import dataclass

import torch
from torch import nn

from cusrl.module.module import Module, ModuleFactory, ModuleFactoryLike
from cusrl.module.normalizer import RunningMeanStd
from cusrl.utils import make_distributed
from cusrl.utils.dict_utils import prefix_dict_keys
from cusrl.utils.typing import Memory, Slice

__all__ = ["Value"]


@dataclass(slots=True)
class ValueFactory(ModuleFactory["Value"]):
    backbone_factory: ModuleFactoryLike
    latent_dim: int | None = None
    action_aware: bool = False

    def __call__(self, input_dim: int | None, output_dim: int = 1) -> "Value":
        return Value(
            self.backbone_factory(input_dim, self.latent_dim),
            value_dim=output_dim,
            action_aware=self.action_aware,
        )


class Value(Module):
    """A module for computing a state-value or action-value function.

    This module uses a backbone network to extract features from the input state
    (and optionally action), and then a linear head to compute the final value.
    It can be configured to be state-dependent or action-dependent (Q-function).

    Args:
        backbone (Module):
            The network used to extract features from the input.
        value_dim (int, optional):
            The dimension of the output value. Defaults to 1.
        action_aware (bool, optional):
            If True, the module becomes an action-value function (Q-function) by
            concatenating the state and action before passing them to the
            backbone. Defaults to False.
    """

    Factory = ValueFactory

    def __init__(
        self,
        backbone: Module,
        value_dim: int = 1,
        action_aware: bool = False,
    ):
        super().__init__(
            input_dim=backbone.input_dim,
            output_dim=value_dim,
            is_recurrent=backbone.is_recurrent,
        )
        self.backbone: Module = backbone.rnn_compatible()
        self.value_head = nn.Linear(backbone.output_dim, value_dim)
        self.value_rms: RunningMeanStd | None = None
        self.action_aware: bool = action_aware

    def to_distributed(self):
        if not self.is_distributed:
            self.is_distributed = True
            self.backbone = self.backbone.to_distributed()
            self.value_head = make_distributed(self.value_head)
        return self

    def evaluate(
        self,
        state: torch.Tensor,
        *,
        action: torch.Tensor | None = None,
        memory: Memory = None,
        done: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        value, _ = self(state, action=action, memory=memory, done=done, **kwargs)
        return value

    def forward(
        self,
        state: torch.Tensor,
        *,
        action: torch.Tensor | None = None,
        memory: Memory = None,
        done: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Memory]:
        if self.action_aware:
            if action is None:
                raise ValueError("Action must be provided when 'action_aware' is True.")
            state = torch.cat([state, action], dim=-1)
        latent, memory = self.backbone(state, memory=memory, done=done, **kwargs)
        self.intermediate_repr["backbone.output"] = latent
        self.intermediate_repr.update(prefix_dict_keys(self.backbone.intermediate_repr, "backbone."))
        return self.value_head(latent), memory

    def step_memory(self, state, memory=None, **kwargs):
        return self.backbone.step_memory(state, memory, **kwargs)

    def reset_memory(self, memory: Memory, done: Slice | torch.Tensor | None = None):
        return self.backbone.reset_memory(memory, done=done)
