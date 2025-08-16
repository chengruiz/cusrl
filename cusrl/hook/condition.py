from collections.abc import Callable, Iterable
from typing import Any

from cusrl.template import ActorCritic, Hook

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

    def __call__(self, agent: ActorCritic, batch) -> bool:
        return batch["epoch_index"] in self.epoch_index


class ConditionalObjectiveActivation(Hook[ActorCritic]):
    """Activates other objective hooks based on specified conditions.

    This hook must be placed before any objective hooks it controls.

    Args:
        named_conditions (Callable[[ActorCritic, dict[str, Any]], bool]):
            Keyword arguments mapping the name of an objective hook to a
            callable condition. The condition determines whether the
            corresponding hook should be active. It receives the agent and the
            current batch and returns `True` if the hook should be active,
            `False` otherwise.
    """

    def __init__(self, **named_conditions: Callable[[ActorCritic, dict[str, Any]], bool]):
        super().__init__()
        self.named_conditions = named_conditions
        self.named_activation = {}

    def pre_update(self, buffer):
        # Store the current activation state of the hooks
        for name in self.named_conditions:
            self.named_activation[name] = self.agent.hook[name].active

    def objective(self, batch) -> None:
        for name, condition in self.named_conditions.items():
            self.agent.hook[name].active_(self.named_activation[name] and condition(self.agent, batch))

    def post_update(self):
        # Restore the activation state of the hooks
        for name in self.named_conditions:
            self.agent.hook[name].active_(self.named_activation[name])
