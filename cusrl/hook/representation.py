from typing import cast

import torch
from torch import nn

from cusrl.module import GraphBuilder, LayerFactoryLike
from cusrl.template import ActorCritic, Hook
from cusrl.utils.typing import Slice

__all__ = ["ReturnPrediction", "StatePrediction", "NextStatePrediction"]


class ReturnPrediction(Hook[ActorCritic]):
    def __init__(
        self,
        latent_name: str = "backbone.output",
        weight: float = 0.01,
        predictor_factory: LayerFactoryLike = nn.Linear,  # type: ignore[assignment]
        predicts_value_instead_of_return: bool = False,
    ):
        super().__init__()
        self.latent_name = latent_name
        self.predictor_factory = predictor_factory
        self.predicts_value_instead_of_return = predicts_value_instead_of_return

        # Mutable attributes
        self.weight: float = weight
        self.register_mutable("weight")

        # Runtime attributes
        self.predictor: nn.Module
        self.criterion: nn.MSELoss

    def init(self):
        self.agent.actor(torch.zeros(1, self.agent.observation_dim))
        latent_dim = self.agent.actor.intermediate_repr[self.latent_name].numel()
        self.register_module("predictor", self.predictor_factory(latent_dim, self.agent.value_dim))
        self.criterion = nn.MSELoss()

    def objective(self, batch):
        latent = self.agent.actor.intermediate_repr[self.latent_name]
        target = batch["value"] if self.predicts_value_instead_of_return else batch["return"]
        with self.agent.autocast():
            prediction = self.predictor(latent)
            return_prediction_loss = self.weight * self.criterion(prediction, target)
        self.agent.record(return_prediction_loss=return_prediction_loss)
        return return_prediction_loss

    def post_export(self, graph: GraphBuilder):
        graph.add_node(
            self.predictor,
            module_name="return_predictor",
            input_names={"input": f"actor.{self.latent_name}"},
            output_names="return_prediction",
            expose_outputs=True,
        )


class StatePrediction(Hook[ActorCritic]):
    """A hook to predict states from actor representations.

    This hook adds an auxiliary loss to the main training objective. The loss is
    the MSELoss between the predicted state and the actual state. This
    encourages the specified latent representation to encode information
    about states (or privileged information).

    Args:
        target_indices (Slice):
            Indices of the state to be predicted.
        latent_name (str):
            The name of the intermediate representation from the actor to be
            used as input of the predictor. Defaults to ``"backbone.output"``.
        weight (float):
            The weight of the state prediction loss in the total objective.
            Defaults to ``0.01``.
        predictor_factory (LayerFactoryLike):
            A callable that creates the predictor module (e.g., `nn.Linear`).
            Defaults to ``nn.Linear``.
    """

    def __init__(
        self,
        target_indices: Slice,
        latent_name: str = "backbone.output",
        weight: float = 0.01,
        predictor_factory: LayerFactoryLike = nn.Linear,  # type: ignore[assignment]
    ):
        super().__init__()
        self.target_indices = target_indices
        self.latent_name = latent_name
        self.predictor_factory = predictor_factory

        # Mutable attributes
        self.weight: float = weight
        self.register_mutable("weight")

        # Runtime attributes
        self.predictor: nn.Module
        self.criterion: nn.MSELoss

    def init(self):
        if not self.agent.has_state:
            raise ValueError("StatePrediction: State is not defined for the agent.")
        self.agent.actor(torch.zeros(1, self.agent.observation_dim))
        latent_dim = self.agent.actor.intermediate_repr[self.latent_name].numel()
        target_dim = torch.zeros(self.agent.state_dim)[self.target_indices].numel()
        self.register_module("predictor", self.predictor_factory(latent_dim, target_dim))
        self.criterion = nn.MSELoss()

    def objective(self, batch):
        state = cast(torch.Tensor, batch["state"])
        with self.agent.autocast():
            latent = self.agent.actor.intermediate_repr[self.latent_name]
            target = state[..., self.target_indices]
            state_prediction_loss = self.weight * self.criterion(self.predictor(latent), target)
        self.agent.record(state_prediction_loss=state_prediction_loss)
        return state_prediction_loss

    def post_export(self, graph: GraphBuilder):
        graph.add_node(
            self.predictor,
            module_name="state_predictor",
            input_names={"input": f"actor.{self.latent_name}"},
            output_names="state_prediction",
            expose_outputs=True,
        )


class ActionAwarePredictorWrapper(nn.Module):
    def __init__(self, wrapped: nn.Module):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, latent: torch.Tensor, action: torch.Tensor | None = None):
        if action is not None:
            latent = torch.cat([latent, action], dim=-1)
        return self.wrapped(latent)


class NextStatePrediction(Hook[ActorCritic]):
    def __init__(
        self,
        target_indices: Slice,
        latent_name: str = "backbone.output",
        weight: float = 0.01,
        predictor_factory: LayerFactoryLike = nn.Linear,  # type: ignore[assignment]
    ):
        super().__init__()
        self.target_indices = target_indices
        self.latent_name = latent_name
        self.predictor_factory = predictor_factory

        # Mutable attributes
        self.weight: float = weight
        self.register_mutable("weight")

        # Runtime attributes
        self.predictor: nn.Module
        self.criterion: nn.MSELoss

    def init(self):
        if not self.agent.has_state:
            raise ValueError("NextStatePrediction: State is not defined for the agent.")
        self.agent.actor(torch.zeros(1, self.agent.observation_dim))
        latent_dim = self.agent.actor.intermediate_repr[self.latent_name].numel()
        target_dim = torch.zeros(self.agent.state_dim)[self.target_indices].numel()
        predictor = self.predictor_factory(latent_dim + self.agent.action_dim, target_dim)
        self.register_module("predictor", ActionAwarePredictorWrapper(predictor))
        self.criterion = nn.MSELoss()

    def objective(self, batch):
        next_state = cast(torch.Tensor, batch["next_state"])
        with self.agent.autocast():
            latent = self.agent.actor.intermediate_repr[self.latent_name]
            target = next_state[..., self.target_indices]
            prediction = self.predictor(latent, batch["action"])
            next_state_prediction_loss = self.weight * self.criterion(prediction, target)
        self.agent.record(next_state_prediction_loss=next_state_prediction_loss)
        return next_state_prediction_loss

    def post_export(self, graph: GraphBuilder):
        graph.add_node(
            self.predictor,
            module_name="next_state_predictor",
            input_names={"latent": f"actor.{self.latent_name}", "action": "action"},
            output_names="next_state_prediction",
            expose_outputs=True,
        )
