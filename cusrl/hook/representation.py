import torch
from torch import nn

from cusrl.module import GraphBuilder, LayerFactoryLike
from cusrl.template import ActorCritic, Hook
from cusrl.utils.typing import Slice

__all__ = ["ReturnPrediction", "StatePrediction", "NextStatePrediction"]


class ReturnPrediction(Hook[ActorCritic]):
    predictor: nn.Module
    criterion: nn.MSELoss

    def __init__(
        self,
        weight: float = 0.01,
        predictor_factory: LayerFactoryLike = nn.Linear,
        predicts_value_instead_of_return: bool = False,
    ):
        super().__init__()
        self.weight = weight
        self.predictor_factory = predictor_factory
        self.predicts_value_instead_of_return = predicts_value_instead_of_return

    def init(self):
        self.register_module("predictor", self.predictor_factory(self.agent.actor.latent_dim, 1))
        self.criterion = nn.MSELoss()

    def objective(self, batch):
        latent = self.agent.actor.intermediate_repr["backbone.output"]
        target = batch["value"] if self.predicts_value_instead_of_return else batch["return"]
        with self.agent.autocast():
            prediction = self.predictor(latent)
            return_prediction_loss = self.weight * self.criterion(prediction, target)
        self.agent.record(return_prediction_loss=return_prediction_loss)
        return return_prediction_loss

    def post_export(self, graph: GraphBuilder):
        graph.add_module_to_graph(
            self.predictor,
            module_name="return_predictor",
            input_names={"input": "actor.backbone.output"},
            output_names="return_prediction",
            expose_outputs=True,
        )


class StatePrediction(Hook[ActorCritic]):
    predictor: nn.Module
    criterion: nn.MSELoss

    def __init__(
        self,
        target_indices: Slice,
        weight: float = 0.01,
        predictor_factory: LayerFactoryLike = nn.Linear,
    ):
        super().__init__()
        self.target_indices = target_indices
        self.weight = weight
        self.predictor_factory = predictor_factory

    def init(self):
        if not self.agent.has_state:
            raise ValueError("StatePrediction: State is not defined for the agent.")
        target_dim = torch.zeros(self.agent.state_dim)[self.target_indices].numel()
        self.register_module("predictor", self.predictor_factory(self.agent.actor.latent_dim, target_dim))
        self.criterion = nn.MSELoss()

    def objective(self, batch):
        with self.agent.autocast():
            latent = self.agent.actor.intermediate_repr["backbone.output"]
            target = batch["state"][..., self.target_indices]
            state_prediction_loss = self.weight * self.criterion(self.predictor(latent), target)
        self.agent.record(state_prediction_loss=state_prediction_loss)
        return state_prediction_loss

    def post_export(self, graph: GraphBuilder):
        graph.add_module_to_graph(
            self.predictor,
            module_name="state_predictor",
            input_names={"input": "actor.backbone.output"},
            output_names="state_prediction",
            expose_outputs=True,
        )


class ActionAwarePredictor(nn.Module):
    def __init__(self, wrapped: nn.Module):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, latent: torch.Tensor, action: torch.Tensor | None = None):
        if action is not None:
            latent = torch.cat([latent, action], dim=-1)
        return self.wrapped(latent)


class NextStatePrediction(Hook[ActorCritic]):
    predictor: nn.Module
    criterion: nn.MSELoss

    def __init__(
        self,
        target_indices: Slice,
        weight: float = 0.01,
        predictor_factory: LayerFactoryLike = nn.Linear,
    ):
        super().__init__()
        self.target_indices = target_indices
        self.weight = weight
        self.predictor_factory = predictor_factory

    def init(self):
        if not self.agent.has_state:
            raise ValueError("NextStatePrediction: State is not defined for the agent.")
        target_dim = torch.zeros(self.agent.state_dim)[self.target_indices].numel()
        predictor = self.predictor_factory(self.agent.actor.latent_dim + self.agent.action_dim, target_dim)
        self.register_module("predictor", ActionAwarePredictor(predictor))
        self.criterion = nn.MSELoss()

    def objective(self, batch):
        with self.agent.autocast():
            latent = self.agent.actor.intermediate_repr["backbone.output"]
            target = batch["next_state"][..., self.target_indices]
            prediction = self.predictor(latent, batch["action"])
            next_state_prediction_loss = self.weight * self.criterion(prediction, target)
        self.agent.record(next_state_prediction_loss=next_state_prediction_loss)
        return next_state_prediction_loss

    def post_export(self, graph: GraphBuilder):
        graph.add_module_to_graph(
            self.predictor,
            module_name="next_state_predictor",
            input_names={"latent": "actor.backbone.output", "action": "action"},
            output_names="next_state_prediction",
            expose_outputs=True,
        )
