from typing import Any

import torch
from torch import nn

from cusrl.module import LayerFactoryLike, Module
from cusrl.template import ActorCritic, Hook
from cusrl.utils.export import ExportSpec
from cusrl.utils.typing import Slice

__all__ = ["ReturnPrediction", "StatePrediction", "NextStatePrediction"]


class ReturnPrediction(Hook[ActorCritic]):
    MODULES = ["predictor"]
    predictor: nn.Module

    def __init__(
        self,
        weight: float = 0.01,
        predictor_factory: LayerFactoryLike = nn.Linear,
        predicts_value_instead_of_return: bool = False,
    ):
        self.weight = weight
        self.predictor_factory = predictor_factory
        self.predicts_value_instead_of_return = predicts_value_instead_of_return
        self.criterion = nn.MSELoss()

    def init(self):
        self.predictor = self.predictor_factory(self.agent.actor.latent_dim, 1)
        self.predictor = self.agent.setup_module(self.predictor)

    def objective(self, batch: dict[str, Any]):
        latent = self.agent.actor.intermediate_repr["backbone.output"]
        target = batch["value"] if self.predicts_value_instead_of_return else batch["return"]
        with self.agent.autocast():
            prediction = self.predictor(latent)
            return_prediction_loss = self.weight * self.criterion(prediction, target)
        self.agent.record(return_prediction_loss=return_prediction_loss)
        return return_prediction_loss

    def export(self, export_data: dict[str, ExportSpec]):
        export_data["actor"].module.return_predictor = self.predictor
        export_data["actor"].module.register_forward_hook(self.__prediction_forward_hook)
        export_data["actor"].output_names.append("return_prediction")

    @staticmethod
    def __prediction_forward_hook(module: Module, args: tuple[Any, ...], output: Any) -> Any:
        return *output, module.return_predictor(module.intermediate_repr["backbone.output"])


class StatePrediction(Hook[ActorCritic]):
    MODULES = ["predictor"]
    predictor: nn.Module

    def __init__(
        self,
        target_indices: Slice,
        weight: float = 0.01,
        predictor_factory: LayerFactoryLike = nn.Linear,
    ):
        self.target_indices = target_indices
        self.weight = weight
        self.predictor_factory = predictor_factory
        self.criterion = nn.MSELoss()

    def init(self):
        if not self.agent.has_state:
            raise ValueError("StatePrediction: State is not defined for the agent.")
        target_dim = torch.zeros(self.agent.state_dim)[self.target_indices].numel()
        self.predictor = self.predictor_factory(self.agent.actor.latent_dim, target_dim)
        self.predictor = self.agent.setup_module(self.predictor)

    def objective(self, batch: dict[str, Any]):
        with self.agent.autocast():
            latent = self.agent.actor.intermediate_repr["backbone.output"]
            target = batch["state"][..., self.target_indices]
            state_prediction_loss = self.weight * self.criterion(self.predictor(latent), target)
        self.agent.record(state_prediction_loss=state_prediction_loss)
        return state_prediction_loss

    def export(self, export_data: dict[str, ExportSpec]):
        export_data["actor"].module.state_predictor = self.predictor
        export_data["actor"].module.register_forward_hook(self.__prediction_forward_hook)
        export_data["actor"].output_names.append("state_prediction")

    @staticmethod
    def __prediction_forward_hook(module: Module, args: tuple[Any, ...], output: Any) -> Any:
        return *output, module.state_predictor(module.intermediate_repr["backbone.output"])


class NextStatePrediction(Hook[ActorCritic]):
    MODULES = ["predictor"]
    predictor: nn.Module

    def __init__(
        self,
        target_indices: Slice,
        weight: float = 0.01,
        predictor_factory: LayerFactoryLike = nn.Linear,
    ):
        self.target_indices = target_indices
        self.weight = weight
        self.predictor_factory = predictor_factory
        self.criterion = nn.MSELoss()

    def init(self):
        if not self.agent.has_state:
            raise ValueError("NextStatePrediction: State is not defined for the agent.")
        target_dim = torch.zeros(self.agent.state_dim)[self.target_indices].numel()
        self.predictor = self.predictor_factory(self.agent.actor.latent_dim + self.agent.action_dim, target_dim)
        self.predictor = self.agent.setup_module(self.predictor)

    def objective(self, batch: dict[str, Any]):
        with self.agent.autocast():
            latent = self.agent.actor.intermediate_repr["backbone.output"]
            target = batch["next_state"][..., self.target_indices]
            prediction = self.predictor(torch.cat([latent, batch["action"]], dim=-1))
            next_state_prediction_loss = self.weight * self.criterion(prediction, target)
        self.agent.record(next_state_prediction_loss=next_state_prediction_loss)
        return next_state_prediction_loss

    def export(self, export_data: dict[str, ExportSpec]):
        export_data["actor"].module.next_state_predictor = self.predictor
        export_data["actor"].module.register_forward_hook(self.__prediction_forward_hook)
        export_data["actor"].output_names.append("next_state_prediction")

    @staticmethod
    def __prediction_forward_hook(module: Module, args: tuple[Any, ...], output: Any) -> Any:
        action = output[0]
        predictor_input = torch.cat([module.intermediate_repr["backbone.output"], action], dim=-1)
        return *output, module.next_state_predictor(predictor_input)
