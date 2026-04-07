from collections.abc import Callable

from cusrl.template import ActorCritic, Hook

__all__ = [
    "OnPolicyBufferCapacitySchedule",
]


class OnPolicyBufferCapacitySchedule(Hook[ActorCritic]):
    """Schedules the capacity of a rollout buffer for an on-policy agent.

    This hook uses a user-provided schedule function that maps the current
    training iteration to the desired number of environment steps per update
    (i.e., rollout length) and resizes the agent's buffer accordingly.

    Args:
        schedule (Callable[[int], int]):
            Function that takes the current iteration index and returns the
            desired buffer capacity. This typically controls the rollout length
            before each iteration.
    """

    def __init__(self, schedule: Callable[[int], int]):
        super().__init__(training_only=True)
        self.schedule = schedule

    def apply_schedule(self, iteration: int):
        capacity = self.schedule(iteration)
        self.agent.num_steps_per_update = capacity
        self.agent.resize_buffer(capacity)
