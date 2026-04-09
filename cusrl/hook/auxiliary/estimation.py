from typing import cast

import torch
from torch import Tensor, nn

from cusrl.nn import Module, ModuleFactoryLike
from cusrl.template import ActorCritic, Hook
from cusrl.utils.typing import Memory, Slice

__all__ = ["StateEstimation"]


class StateEstimation(Hook[ActorCritic]):
    """Learns an auxiliary estimator between transition entries.

    This hook runs a dedicated estimator on a selected slice of one transition
    entry, stores the prediction in the rollout transition, and optimizes the
    estimator with an MSE loss against a selected slice of another entry. It is
    typically used to reconstruct privileged state information from
    observations, but it can target any pair of transition tensors.

    The estimator is made RNN-compatible and its memory is threaded through
    rollout collection via the ``"estimator_memory"`` transition key so the
    same module can be trained on sequential batches.

    Args:
        estimator_factory (ModuleFactoryLike):
            Factory used to build the estimator module. It receives the sliced
            source and target dimensions and must return a module compatible
            with :class:`cusrl.nn.Module`.
        source_name (str):
            Transition key used as the estimator input. Defaults to
            ``"observation"``.
        source_indices (Slice):
            Slice applied to ``source_name`` before passing it to the
            estimator. Defaults to ``slice(None)``.
        source_dim (int | None):
            Full dimension of ``source_name`` before slicing. When omitted, it
            is inferred for ``"observation"``, ``"next_observation"``,
            ``"state"``, and ``"next_state"``. Defaults to ``None``.
        target_name (str):
            Transition key used as the supervision target. Defaults to
            ``"state"``.
        target_indices (Slice):
            Slice applied to ``target_name`` before computing the loss.
            Defaults to ``slice(None)``.
        target_dim (int | None):
            Full dimension of ``target_name`` before slicing. When omitted, it
            is inferred for ``"observation"``, ``"next_observation"``,
            ``"state"``, and ``"next_state"``. Defaults to ``None``.
        estimation_name (str):
            Transition key used to store the estimator output during rollout.
            Defaults to ``"state_estimation"``.
        weight (float):
            Multiplicative weight applied to the state estimation loss in the
            returned objective dictionary. Defaults to ``1.0``.
    """

    def __init__(
        self,
        estimator_factory: ModuleFactoryLike,
        source_name: str = "observation",
        source_indices: Slice = slice(None),
        source_dim: int | None = None,
        target_name: str = "state",
        target_indices: Slice = slice(None),
        target_dim: int | None = None,
        estimation_name: str = "state_estimation",
        weight: float = 1.0,
    ):
        super().__init__()
        self.estimator_factory = estimator_factory
        self.source_name = source_name
        self.source_indices = source_indices
        self.source_dim = source_dim
        self.target_name = target_name
        self.target_indices = target_indices
        self.target_dim = target_dim
        self.estimation_name = estimation_name

        # Mutable attributes
        self.weight: float = weight
        self.register_mutable("weight")

        # Runtime attributes
        self.estimator: Module
        self.criterion: nn.MSELoss
        self._estimator_memory: Memory = None

    def init(self):
        if self.source_dim is None:
            if self.source_name == "observation" or self.source_name == "next_observation":
                self.source_dim = self.agent.observation_dim
            elif self.source_name == "state" or self.source_name == "next_state":
                self.source_dim = self.agent.state_dim
            else:
                raise ValueError(f"'source_dim' must be specified for source_name '{self.source_name}'.")
        if self.target_dim is None:
            if self.target_name == "observation" or self.target_name == "next_observation":
                self.target_dim = self.agent.observation_dim
            elif self.target_name == "state" or self.target_name == "next_state":
                self.target_dim = self.agent.state_dim
            else:
                raise ValueError(f"'target_dim' must be specified for target_name '{self.target_name}'.")

        source_dim = torch.zeros(1, self.source_dim)[..., self.source_indices].numel()
        target_dim = torch.zeros(1, self.target_dim)[..., self.target_indices].numel()
        self.register_module("estimator", self.estimator_factory(source_dim, target_dim).rnn_compatible())
        self.criterion = nn.MSELoss()

    def pre_act(self, transition):
        source = cast(Tensor, transition[self.source_name])[..., self.source_indices]
        estimation, next_estimator_memory = self.estimator(
            source,
            memory=self._estimator_memory,
            sequential=False,
        )

        transition[self.estimation_name] = estimation
        transition["estimator_memory"] = self._estimator_memory
        self._estimator_memory = next_estimator_memory

    def post_step(self, transition):
        self.estimator.reset_memory(self._estimator_memory, cast(Tensor, transition["done"]))

    def objective(self, metadata, batch):
        source = cast(Tensor, batch[self.source_name])[..., self.source_indices]
        target = cast(Tensor, batch[self.target_name])[..., self.target_indices]
        estimation, _ = self.estimator(source, memory=batch["estimator_memory"], done=batch["done"])
        state_estimation_loss = self.criterion(estimation, target)
        return {"state_estimation_loss": state_estimation_loss * self.weight}
