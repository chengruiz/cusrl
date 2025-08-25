from typing import Any

__all__ = [
    "LessThan",
    "NotLessThan",
    "PiecewiseLinearFunction",
    "StepFunction",
]


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


class StepFunction:
    """A step function.

    The function starts with an initial value and changes its value at specific
    points. The points must be sorted by their x-coordinate in increasing order.

    Args:
        initial_value (Any):
            The initial value of the function.
        *points (tuple[int, Any]):
            A sequence of points (x, y) where the function value changes. At
            each point, for an iteration >= x, the function value becomes y. The
            x-coordinates of the points must be strictly increasing.
    """

    def __init__(self, initial_value: Any, *points: tuple[int, Any]):
        self.initial_value = initial_value
        self.points = points
        if any(x0 >= x1 for (x0, _), (x1, _) in zip(self.points, self.points[1:])):
            raise ValueError("X coordinates must be strictly increasing")

    def __call__(self, iteration: int) -> Any:
        value = self.initial_value
        for x, y in self.points:
            if iteration < x:
                break
            value = y
        return value


class PiecewiseLinearFunction:
    """A piecewise linear function.

    The function is defined by a set of points. It linearly interpolates between
    consecutive points. Before the first point, it returns the y-value of the
    first point. After the last point, it returns the y-value of the last point.

    Args:
        point1 (tuple[int, float]):
            The first point (x, y).
        point2 (tuple[int, float]):
            The second point (x, y).
        *points (tuple[int, float]):
            Additional points (x, y). The x-coordinates of all points must be
            strictly increasing.
    """

    def __init__(self, point1: tuple[int, float], point2: tuple[int, float], *points: tuple[int, float]):
        self.points = (point1, point2, *points)
        if any(x0 >= x1 for (x0, _), (x1, _) in zip(self.points, self.points[1:])):
            raise ValueError("X coordinates must be strictly increasing")

    def __call__(self, iteration: int) -> float:
        # Left of first point
        if iteration <= self.points[0][0]:
            return self.points[0][1]
        # Interpolate within range
        for (x0, y0), (x1, y1) in zip(self.points, self.points[1:]):
            if iteration <= x1:
                return y0 + (y1 - y0) * (iteration - x0) / (x1 - x0)
        # Right of last point
        return self.points[-1][1]
