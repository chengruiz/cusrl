from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from cusrl.template import ActorCritic, Hook, HookFactory

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

    def __call__(self, agent: ActorCritic, metadata: dict[str, Any], batch: dict[str, Any]) -> bool:
        return metadata["epoch_index"] in self.epoch_index


class ActivationCondition(Protocol):
    def __call__(self, agent: ActorCritic, metadata: dict[str, Any], batch: dict[str, Any]) -> bool:
        """Determines whether a hook should be active based on the agent's state
        and the current batch."""


class ConditionalObjectiveActivation(Hook[ActorCritic]):
    """Activates other objective hooks based on specified conditions.

    This hook must be placed before any objective hooks it controls.

    Args:
        named_conditions (Callable[[ActorCritic, dict[str, Any], dict[str, Any]], bool]):
            Keyword arguments mapping the name of an objective hook to a
            callable condition. The condition determines whether the
            corresponding hook should be active. It receives the agent and the
            current metadata and batch and returns ``True`` if the hook should
            be active, ``False`` otherwise.
    """

    @dataclass
    class Factory(HookFactory["ConditionalObjectiveActivation"]):
        named_conditions: dict[str, ActivationCondition] = field(default_factory=dict)

        @classmethod
        def get_hook_type(cls):
            return ConditionalObjectiveActivation

    def __init__(
        self,
        named_conditions: dict[str, ActivationCondition] | None = None,
        **kwargs: ActivationCondition,
    ):
        super().__init__(training_only=True)
        self.named_conditions = (named_conditions or {}) | kwargs
        self._named_activation = {}

    def pre_update(self, buffer):
        # Store the current activation state of the hooks
        for name in self.named_conditions:
            self._named_activation[name] = self.agent.hook[name].active

    def pre_objective(self, metadata, batch):
        for name, condition in self.named_conditions.items():
            is_active = condition(self.agent, metadata, batch)
            self.agent.hook[name].active_(self._named_activation[name] and is_active)

    def post_update(self):
        # Restore the activation state of the hooks
        for name in self.named_conditions:
            self.agent.hook[name].active_(self._named_activation[name])
