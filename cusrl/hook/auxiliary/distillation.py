from typing import cast

import torch
from torch import nn

from cusrl.nn.module.distribution import MeanStdDict
from cusrl.template import ActorCritic, Hook

__all__ = ["PolicyDistillation", "PolicyDistillationLoss"]


class PolicyDistillationLoss(Hook[ActorCritic]):
    """Matches the policy mean against precomputed expert actions.

    This hook assumes another component has already written expert actions into
    each transition or training batch. During optimization it compares the
    current policy mean with that target action tensor using an MSE loss.

    Args:
        target_name (str):
            Transition or batch key containing the expert action targets.
            Defaults to ``"expert_action"``.
        weight (float):
            Multiplicative weight applied to the distillation loss. Defaults to
            ``1.0``.
    """

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

    def objective(self, metadata, batch):
        action_dist = cast(MeanStdDict, batch["curr_action_dist"])
        distillation_loss = self.criterion(action_dist["mean"], batch[self.target_name])
        return {"distillation_loss": distillation_loss * self.weight}


class PolicyDistillation(PolicyDistillationLoss):
    """Distills a pre-trained expert.

    This hook loads an exported TorchScript expert, queries it on each rollout
    step, and stores the resulting actions under ``target_name`` inherited from
    :class:`PolicyDistillationLoss`. The inherited objective then trains the
    current policy to match those expert actions with an MSE loss on the policy
    mean.

    The expert is treated as a recurrent policy and is reset with the
    transition ``"done"`` mask after each environment step.

    Args:
        expert_path (str):
            Path to the exported TorchScript expert policy.
        observation_name (str):
            Transition key used as input to the expert policy. Defaults to
            ``"observation"``.
        weight (float):
            Multiplicative weight applied to the inherited distillation loss.
            Defaults to ``1.0``.
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
            raise ValueError("'expert_path' cannot be empty")

        self.expert = torch.jit.load(self.expert_path, map_location=self.agent.device)

    @torch.no_grad()
    def post_step(self, transition):
        transition[self.target_name] = self.expert(transition[self.observation_name])
        if hasattr(self.expert, "reset"):
            self.expert.reset(transition["done"].squeeze(-1))
