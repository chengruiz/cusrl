import itertools
from typing import Any, cast

import torch
from torch import nn

from cusrl.module import Module, ModuleFactoryLike
from cusrl.template import Buffer, Hook
from cusrl.utils.helper import get_or
from cusrl.utils.typing import Slice

__all__ = ["RandomNetworkDistillation"]


class RandomNetworkDistillation(Hook):
    """A hook to generate intrinsic rewards with Random Network Distillation (RND).

    This method is described in "Exploration by Random Network Distillation",
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
    MODULES = ["target", "predictor"]
    MUTABLE_ATTRS = ["reward_scale"]

    def __init__(
        self,
        module_factory: ModuleFactoryLike,
        output_dim: int,
        reward_scale: float,
        state_indices: Slice | None = None,
    ):
        self.output_dim = output_dim
        self.reward_scale = reward_scale
        self.module_factory = module_factory
        self.state_indices = slice(None) if state_indices is None else state_indices
        self.criterion = nn.MSELoss()

    def init(self):
        input_dim = torch.ones(1, self.agent.state_dim)[..., self.state_indices].numel()
        self.target = self.module_factory(input_dim, self.output_dim)
        self.predictor = self.module_factory(input_dim, self.output_dim)

        for module in itertools.chain(self.target.modules(), self.predictor.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        self.target = self.agent.setup_module(self.target)
        self.predictor = self.agent.setup_module(self.predictor)
        self.target.requires_grad_(False)

    @torch.no_grad()
    def pre_update(self, buffer: Buffer):
        next_state = cast(torch.Tensor, get_or(buffer, "next_state", "next_observation"))[..., self.state_indices]
        target, prediction = self.target(next_state), self.predictor(next_state)
        rnd_reward = self.reward_scale * (target - prediction).square().mean(dim=-1, keepdim=True)
        cast(torch.Tensor, buffer["reward"]).add_(rnd_reward)
        self.agent.record(rnd_reward=rnd_reward)

    def objective(self, batch: dict[str, Any]):
        with self.agent.autocast():
            state = get_or(batch, "state", "observation")[..., self.state_indices]
            rnd_loss = self.criterion(self.predictor(state), self.target(state))
        self.agent.record(rnd_loss=rnd_loss)
        return rnd_loss
