import math
from typing import Any, TypeAlias

__all__ = [
    "CosineAnnealingScheduler",
    "LessThan",
    "NotLessThan",
    "PiecewiseLinearScheduler",
    "StepScheduler",
    "TanhScheduler",
]


Anchor: TypeAlias = tuple[int, float]
Transition: TypeAlias = tuple[int, Any]


def _validate_strictly_increasing_steps(anchors: tuple[Transition, ...]) -> None:
    if any(step0 >= step1 for (step0, _), (step1, _) in zip(anchors, anchors[1:])):
        raise ValueError("step coordinates must be strictly increasing")


class LessThan:
    """A callable that returns True if a value is less than a threshold.

    Args:
        threshold (int): The threshold to compare against.
    """

    def __init__(self, threshold: int):
        self.threshold = threshold

    def __call__(self, value: int) -> bool:
        return value < self.threshold


class NotLessThan:
    """A callable that returns True if a value is not less than a threshold.

    This is equivalent to checking if the value is greater than or equal to
    the threshold.

    Args:
        threshold (int): The threshold to compare against.
    """

    def __init__(self, threshold: int):
        self.threshold = threshold

    def __call__(self, value: int) -> bool:
        return value >= self.threshold


class StepScheduler:
    """A step function scheduler.

    The function starts with an initial value and changes its value at specific
    transitions. The transitions must be sorted by their step in increasing order.

    Args:
        initial_value (Any):
            The initial value of the function.
        *transitions (tuple[int, Any]):
            A sequence of transitions (step, value) where the scheduled value
            changes. At each transition, for an iteration >= step, the scheduled
            value becomes value. The steps must be strictly increasing.
    """

    def __init__(self, initial_value: Any, *transitions: Transition):
        self.initial_value = initial_value
        self.transitions = transitions
        _validate_strictly_increasing_steps(self.transitions)

    def __call__(self, iteration: int) -> Any:
        value = self.initial_value
        for step, scheduled_value in self.transitions:
            if iteration < step:
                break
            value = scheduled_value
        return value


class PiecewiseLinearScheduler:
    """A piecewise linear function scheduler.

    The function is defined by a set of anchors. It linearly interpolates
    between consecutive anchors. Before the first anchor, it returns the first
    value. After the last anchor, it returns the last value.

    Args:
        *anchors (tuple[int, float]):
            A sequence of anchors (step, value). At least two anchors are
            required, and their steps must be strictly increasing.
    """

    def __init__(self, *anchors: Anchor):
        if len(anchors) < 2:
            raise ValueError("at least two anchors are required")
        self.anchors = anchors
        _validate_strictly_increasing_steps(self.anchors)

    def __call__(self, iteration: int) -> float:
        first_step, first_value = self.anchors[0]
        if iteration <= first_step:
            return first_value

        for (start_step, start_value), (end_step, end_value) in zip(self.anchors, self.anchors[1:]):
            if iteration <= end_step:
                ratio = (iteration - start_step) / (end_step - start_step)
                return start_value + (end_value - start_value) * ratio

        return self.anchors[-1][1]


class CosineAnnealingScheduler:
    """A scheduler that interpolates between two values with cosine annealing.

    Args:
        start (tuple[int, float]):
            The start anchor (step, value).
        end (tuple[int, float]):
            The end anchor (step, value).
    """

    def __init__(self, start: Anchor, end: Anchor):
        self.start_step, self.start_value = start
        self.end_step, self.end_value = end
        _validate_strictly_increasing_steps((start, end))

    def __call__(self, iteration: int) -> float:
        if iteration <= self.start_step:
            return self.start_value
        if iteration >= self.end_step:
            return self.end_value

        ratio = (iteration - self.start_step) / (self.end_step - self.start_step)
        return self.end_value + 0.5 * (self.start_value - self.end_value) * (1 + math.cos(math.pi * ratio))


class TanhScheduler:
    """A scheduler that interpolates between two values using a hyperbolic
    tangent function.

    Args:
        start (tuple[int, float]):
            The start anchor (step, value).
        end (tuple[int, float]):
            The end anchor (step, value).
        eta (float):
            A positive parameter that controls the steepness of the transition.
    """

    def __init__(self, start: Anchor, end: Anchor, eta: float):
        self.start_step, self.start_value = start
        self.end_step, self.end_value = end
        self.eta = eta
        _validate_strictly_increasing_steps((start, end))
        if self.eta <= 0:
            raise ValueError("'eta' must be positive")
        self.mid_step = (self.start_step + self.end_step) / 2
        self.start_epsilon = self._get_epsilon(self.start_step)
        self.end_epsilon = self._get_epsilon(self.end_step)

    def _get_epsilon(self, iteration: int) -> float:
        ratio = 2 * (iteration - self.mid_step) / (self.end_step - self.start_step)
        return 0.5 + 0.5 * math.tanh(self.eta * ratio)

    def __call__(self, iteration: int) -> float:
        if iteration <= self.start_step:
            return self.start_value
        if iteration >= self.end_step:
            return self.end_value

        ratio = (self._get_epsilon(iteration) - self.start_epsilon) / (self.end_epsilon - self.start_epsilon)
        return self.start_value + (self.end_value - self.start_value) * ratio
