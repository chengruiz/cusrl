from typing import cast

import torch
from torch import Tensor, nn

from cusrl.module.distribution import MeanStdDict
from cusrl.template import ActorCritic, Hook

__all__ = ["PolicyDistillation", "PolicyDistillationLoss"]


class PolicyDistillationLoss(Hook[ActorCritic]):
    def __init__(
        self,
        target_name: str = "expert_action",
        weight: float = 1.0,
    ):
        super().__init__()
        self.target_name: str = target_name
        self.weight: float = weight
        self.register_mutable("weight")

        # Runtime attributes
        self.criterion: nn.MSELoss

    def init(self):
        self.criterion = nn.MSELoss()

    def objective(self, batch) -> Tensor:
        action_dist = cast(MeanStdDict, batch["curr_action_dist"])
        distillation_loss = self.criterion(action_dist["mean"], batch[self.target_name]) * self.weight
        self.agent.record(distillation_loss=distillation_loss)
        return distillation_loss


class PolicyDistillation(PolicyDistillationLoss):
    """Distills a pre-trained expert.

    This hook computes a loss that encourages the agent's policy to mimic the
    actions of a pre-trained expert policy.

    Args:
        expert_path (str):
            The file path to the exported TorchScript expert policy.
        observation_name (str, optional):
            The key in the transition dictionary that corresponds to the
            observation tensor. Defaults to `"observation"`.
        weight (float, optional):
            Weight for the distillation loss. Defaults to 1.0.
    """

    def __init__(
        self,
        expert_path: str,
        observation_name: str = "observation",
        weight: float = 1.0,
    ):
        super().__init__(weight=weight)
        self.expert_path = expert_path
        self.observation_name = observation_name

        # Runtime attributes
        self.expert: torch.jit.ScriptModule

    def init(self):
        super().init()
        if not self.expert_path:
            raise ValueError("'expert_path' cannot be empty.")

        self.expert = torch.jit.load(self.expert_path, map_location=self.agent.device)

    @torch.no_grad()
    def post_step(self, transition):
        transition[self.target_name] = self.expert(transition[self.observation_name])
        self.expert.reset(cast(torch.Tensor, transition["done"]).squeeze(-1))
