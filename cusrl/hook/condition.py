from collections.abc import Callable, Iterable
from typing import Any

import torch

from cusrl.template import ActorCritic, Buffer, Hook

__all__ = ["ConditionalObjectiveActivation", "EpochIndexCondition"]


class EpochIndexCondition:
    """Checks if the current epoch index is in a specified set of epoch indices.

    Args:
        epoch_index (int | Iterable[int]):
            A single epoch index or an iterable of epoch indices.
    """

    def __init__(self, epoch_index: int | Iterable[int]):
        if isinstance(epoch_index, int):
            epoch_index = [epoch_index]
        self.epoch_index = set(epoch_index)

    def __call__(self, agent: ActorCritic, batch: dict[str, Any]) -> bool:
        return batch["epoch_index"] in self.epoch_index


class ConditionalObjectiveActivation(Hook[ActorCritic]):
    """A hook to activate other objective hooks based on specified conditions.

    This hook must be placed before any objective hooks it controls.

    Args:
        named_conditions (Callable[[ActorCritic, dict[str, Any]], bool]):
            Keyword arguments mapping the name of an objective hook to a callable
            condition. The condition determines whether the corresponding hook
            should be active. It receives the agent and the current batch and
            returns `True` if the hook should be active, `False` otherwise.
    """

    def __init__(self, **named_conditions: Callable[[ActorCritic, dict[str, Any]], bool]):
        self.named_conditions = named_conditions
        self.named_activation = {}

    def pre_update(self, buffer: "Buffer"):
        # Store the current activation state of the hooks
        for name in self.named_conditions:
            self.named_activation[name] = self.agent.hook[name].active

    def objective(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        for name, condition in self.named_conditions.items():
            self.agent.hook[name].active = self.named_activation[name] and condition(self.agent, batch)

    def post_update(self):
        # Restore the activation state of the hooks
        for name in self.named_conditions:
            self.agent.hook[name].active = self.named_activation[name]
