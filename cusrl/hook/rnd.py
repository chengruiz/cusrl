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
    """Generates intrinsic rewards with Random Network Distillation (RND).

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

    def __init__(
        self,
        module_factory: ModuleFactoryLike,
        output_dim: int,
        reward_scale: float,
        state_indices: Slice | None = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.module_factory = module_factory
        self.state_indices = slice(None) if state_indices is None else state_indices

        # Mutable attributes
        self.reward_scale: float
        self.register_mutable("reward_scale", reward_scale)

        # Runtime attributes
        self.target: Module
        self.predictor: Module
        self.criterion: nn.MSELoss

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
        next_state = cast(torch.Tensor, get_first(buffer, "next_state", "next_observation"))
        rnd_next_state = next_state[..., self.state_indices]
        target, prediction = self.target(rnd_next_state), self.predictor(rnd_next_state)
        rnd_reward = self.reward_scale * (target - prediction).square().mean(dim=-1, keepdim=True)
        cast(torch.Tensor, buffer["reward"]).add_(rnd_reward)
        self.agent.record(rnd_reward=rnd_reward)

    def objective(self, batch):
        state = cast(torch.Tensor, get_first(batch, "state", "observation"))
        rnd_state = state[..., self.state_indices]
        with self.agent.autocast():
            rnd_loss = self.criterion(self.predictor(rnd_state), self.target(rnd_state))
        self.agent.record(rnd_loss=rnd_loss)
        return rnd_loss
