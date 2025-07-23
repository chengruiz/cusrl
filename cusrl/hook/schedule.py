from collections.abc import Callable
from typing import Any

from cusrl.template import ActorCritic, Hook

__all__ = [
    "HookActivationSchedule",
    "LessThan",
    "NotLessThan",
    "OnPolicyBufferCapacitySchedule",
    "ParameterSchedule",
    "PiecewiseFunction",
]


class ParameterSchedule(Hook[ActorCritic]):
    """A hook for scheduling updates to a specific parameter of a hook.

    Args:
        hook_name (str):
            The name of the hook whose parameter will be updated.
        parameter (str):
            The name of the parameter to be updated.
        schedule (Callable[[int], Any]):
            A callable that defines the schedule for updating the parameter.
            It takes the current iteration as input and returns the new value
            for the parameter.
    """

    def __init__(
        self,
        hook_name: str,
        parameter: str,
        schedule: Callable[[int], Any],
    ):
        self.hook_name = hook_name
        self.parameter = parameter
        self.schedule = schedule

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}_{self.hook_name}_{self.parameter}"

    def apply_schedule(self, iteration: int):
        hook = self.agent.hook[self.hook_name]
        hook.update_attribute(self.parameter, self.schedule(iteration))


class HookActivationSchedule(Hook[ActorCritic]):
    """A hook that activates or deactivates a specific hook based on a schedule.

    Args:
        hook_name (str):
            The name of the hook to be activated or deactivated.
        schedule (Callable[[int], bool]):
            A callable that defines the schedule for activating or deactivating
            the hook. It takes the current iteration as input and returns a
            boolean value indicating whether to activate the hook.
    """

    def __init__(
        self,
        hook_name: str,
        schedule: Callable[[int], bool],
    ):
        self.hook_name = hook_name
        self.schedule = schedule

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}_{self.hook_name}"

    def apply_schedule(self, iteration: int):
        self.agent.hook[self.hook_name].active = self.schedule(iteration)


class OnPolicyBufferCapacitySchedule(Hook[ActorCritic]):
    def __init__(self, schedule: Callable[[int], int]):
        self.schedule = schedule

    def apply_schedule(self, iteration: int):
        capacity = self.schedule(iteration)
        self.agent.num_steps_per_update = capacity
        self.agent.resize_buffer(capacity)


class LessThan:
    def __init__(self, threshold: int):
        self.threshold = threshold

    def __call__(self, value: int) -> bool:
        return value < self.threshold


class NotLessThan:
    def __init__(self, threshold: int):
        self.threshold = threshold

    def __call__(self, value: int) -> bool:
        return value >= self.threshold


class PiecewiseFunction:
    def __init__(self, initial_value: Any, *segments: tuple[int, Any]):
        self.initial_value = initial_value
        self.segments = segments

    def __call__(self, iteration: int) -> Any:
        value = self.initial_value
        for threshold, segment_value in self.segments:
            if iteration < threshold:
                return value
            value = segment_value
        return value
