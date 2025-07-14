from dataclasses import dataclass

import torch
from torch import nn

from cusrl.module.module import Module, ModuleFactory

__all__ = ["Simba"]


@dataclass(slots=True)
class SimbaFactory(ModuleFactory["Simba"]):
    hidden_dim: int | None = None
    num_blocks: int = 1
    activation_fn: str | type[nn.Module] = nn.ReLU

    def __call__(self, input_dim: int, output_dim: int | None = None):
        return Simba(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            output_dim=output_dim,
            activation_fn=self._resolve_activation_fn(self.activation_fn),
        )


class SimbaBlock(nn.Sequential):
    def __init__(self, hidden_dim: int, activation_fn: type[nn.Module] = nn.ReLU):
        super().__init__(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            activation_fn(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, input):
        return input + super().forward(input)


class Simba(Module):
    """A network architecture described in:
    "SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning",
    https://arxiv.org/abs/2410.09754
    """

    Factory = SimbaFactory

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        num_blocks: int = 1,
        output_dim: int | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
    ):
        self.hidden_dim = hidden_dim or input_dim
        super().__init__(input_dim, output_dim or self.hidden_dim)
        self.num_blocks = num_blocks

        self.blocks = nn.Sequential()
        if hidden_dim is not None:
            self.blocks.append(nn.Linear(input_dim, self.hidden_dim))
        for i in range(num_blocks):
            self.blocks.append(SimbaBlock(self.hidden_dim, activation_fn))
        self.blocks.append(nn.LayerNorm(self.hidden_dim))
        if output_dim is not None:
            self.blocks.append(nn.Linear(self.hidden_dim, output_dim))

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.blocks(input)
