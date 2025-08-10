import itertools
from typing import cast

import torch
from torch import nn

from cusrl.module import Module, ModuleFactoryLike
from cusrl.template import Buffer, Hook
from cusrl.utils.dict_utils import get_first
from cusrl.utils.typing import Slice

__all__ = ["RandomNetworkDistillation"]


class RandomNetworkDistillation(Hook):
    """A hook to generate intrinsic rewards with Random Network Distillation.

    Described in "Exploration by Random Network Distillation",
    https://arxiv.org/abs/1810.12894

    Args:
        module_factory (ModuleFactoryLike):
            Factory for creating the target and predictor networks.
        output_dim (int):
            Output dimension of the target and predictor networks.
        reward_scale (float):
            The scale of the intrinsic reward.
        state_indices (Slice | None, optional):
            Indices of states used for quantifying novelty. Defaults to None.
    """

    target: Module
    predictor: Module
    criterion: nn.MSELoss

    # Mutable attributes
    reward_scale: float

    def __init__(
        self,
        module_factory: ModuleFactoryLike,
        output_dim: int,
        reward_scale: float,
        state_indices: Slice | None = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.register_mutable("reward_scale", reward_scale)
        self.module_factory = module_factory
        self.state_indices = slice(None) if state_indices is None else state_indices

    def init(self):
        input_dim = torch.ones(1, self.agent.state_dim)[..., self.state_indices].numel()
        target = self.module_factory(input_dim, self.output_dim)
        predictor = self.module_factory(input_dim, self.output_dim)

        for module in itertools.chain(target.modules(), predictor.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        self.register_module("target", target)
        self.register_module("predictor", predictor)
        self.target.requires_grad_(False)
        self.criterion = nn.MSELoss()

    @torch.no_grad()
    def pre_update(self, buffer: Buffer):
        next_state = cast(torch.Tensor, get_first(buffer, "next_state", "next_observation"))[..., self.state_indices]
        target, prediction = self.target(next_state), self.predictor(next_state)
        rnd_reward = self.reward_scale * (target - prediction).square().mean(dim=-1, keepdim=True)
        cast(torch.Tensor, buffer["reward"]).add_(rnd_reward)
        self.agent.record(rnd_reward=rnd_reward)

    def objective(self, batch):
        with self.agent.autocast():
            state = get_first(batch, "state", "observation")[..., self.state_indices]
            rnd_loss = self.criterion(self.predictor(state), self.target(state))
        self.agent.record(rnd_loss=rnd_loss)
        return rnd_loss
