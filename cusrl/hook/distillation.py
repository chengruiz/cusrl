from typing import cast

import torch
from torch import Tensor, nn

from cusrl.module.distribution import MeanStdDict
from cusrl.template import ActorCritic, Hook

__all__ = ["PolicyDistillationLoss"]


class PolicyDistillationLoss(Hook[ActorCritic]):
    """Distills a pre-trained expert.

    This hook computes a loss that encourages the agent's policy to mimic the
    actions of a pre-trained expert policy.

    Args:
        expert_path (str):
            The file path to the exported TorchScript expert policy.
        observation_name (str, optional):
            The key in the transition dictionary that corresponds to the
            observation tensor. Defaults to "observation".
        weight (float, optional):
            Weight for the distillation loss. Defaults to 1.0.
    """

    def __init__(
        self,
        expert_path: str,
        observation_name: str = "observation",
        weight: float = 1.0,
    ):
        super().__init__()
        self.expert_path = expert_path
        self.observation_name = observation_name
        self.weight: float
        self.register_mutable("weight", weight)

        # Runtime attributes
        self.expert: torch.jit.ScriptModule
        self.criterion: nn.MSELoss

    def init(self):
        if not self.expert_path:
            raise ValueError("'expert_path' cannot be empty.")

        self.expert = torch.jit.load(self.expert_path, map_location=self.agent.device)
        self.criterion = nn.MSELoss()

    @torch.no_grad()
    def post_step(self, transition):
        transition["expert_action"] = self.expert(transition[self.observation_name])
        self.expert.reset(cast(torch.Tensor, transition["done"]).squeeze(-1))

    def objective(self, batch) -> Tensor:
        action_dist = cast(MeanStdDict, batch["curr_action_dist"])
        distillation_loss = self.criterion(action_dist["mean"], batch["expert_action"]) * self.weight
        self.agent.record(distillation_loss=distillation_loss)
        return distillation_loss
