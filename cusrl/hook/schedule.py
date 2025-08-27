from collections.abc import Callable
from typing import Any

from cusrl.template import ActorCritic, Hook

__all__ = [
    "HookActivationSchedule",
    "HookParameterSchedule",
]


class HookParameterSchedule(Hook[ActorCritic]):
    """Schedules updates to a specific parameter of a hook.

    Args:
        hook_name (str):
            The name of the hook whose parameter will be updated.
        parameter (str):
            The name of the parameter to be updated.
        scheduler (Callable[[int], Any]):
            A callable that defines the schedule for updating the parameter.
            It takes the current iteration as input and returns the new value
            for the parameter.
    """

    def __init__(
        self,
        hook_name: str,
        parameter: str,
        scheduler: Callable[[int], Any],
    ):
        super().__init__()
        self.hook_name = hook_name
        self.parameter = parameter
        self.scheduler = scheduler
        self.name_(f"{self.hook_name}_{self.parameter}_schedule")

    def apply_schedule(self, iteration: int):
        hook = self.agent.hook[self.hook_name]
        value = self.scheduler(iteration)
        hook.update_attribute(self.parameter, value)
        if isinstance(value, float):
            self.agent.record(**{f"{self.hook_name}_{self.parameter}": value})


class HookActivationSchedule(Hook[ActorCritic]):
    """Activates or deactivates a specific hook based on a schedule.

    Args:
        hook_name (str):
            The name of the hook to be activated or deactivated.
        scheduler (Callable[[int], bool]):
            A callable that defines the schedule for activating or deactivating
            the hook. It takes the current iteration as input and returns a
            boolean value indicating whether to activate the hook.
    """

    def __init__(self, hook_name: str, scheduler: Callable[[int], bool]):
        super().__init__()
        self.hook_name = hook_name
        self.scheduler = scheduler
        self.name_(f"{hook_name}_activation_schedule")

    def apply_schedule(self, iteration: int):
        self.agent.hook[self.hook_name].active_(self.scheduler(iteration))
