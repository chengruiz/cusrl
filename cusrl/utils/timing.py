import time
from collections import defaultdict
from contextlib import contextmanager

__all__ = ["Timer", "Rate"]


class Timer:
    """A utility class for measuring and accumulating execution time of code
    blocks, functions, or named sections."""

    def __init__(self):
        self.start_time = {}
        self.total_time = defaultdict(int)

    def start(self, name):
        if name in self.start_time:
            raise RuntimeError(f"Timer '{name}' already started.")
        self.start_time[name] = time.time()

    def stop(self, name):
        try:
            start_time = self.start_time.pop(name)
        except KeyError as error:
            raise RuntimeError(f"Timer '{name}' not started.") from error
        self.total_time[name] += time.time() - start_time

    def __getitem__(self, item):
        return self.total_time[item]

    def clear(self):
        self.start_time.clear()
        self.total_time.clear()

    def wrap(self, name, func):
        def wrapper(*args, **kwargs):
            self.start(name)
            result = func(*args, **kwargs)
            self.stop(name)
            return result

        return wrapper

    def decorate(self, name):
        def decorator(func):
            return self.wrap(name, func)

        return decorator

    @contextmanager
    def record(self, name):
        self.start(name)
        yield
        self.stop(name)


class Rate:
    """A helper class to run loops at a desired frequency.

    This class is designed to be used in a loop to maintain a specific
    frequency, often referred to as frames per second (fps).

    Args:
        fps: The desired frequency in Hertz (frames per second).
        threshold:
            The maximum time in seconds the internal clock is allowed to fall
            behind real time. This prevents the loop from trying to catch up
            after a long pause or a single very slow iteration by resetting
            the anchor time if it's too far in the past. Defaults to ``0.05``.
    """

    def __init__(self, fps: float, threshold: float = 0.05):
        if fps <= 0:
            raise ValueError("fps must be > 0.")
        if threshold < 0:
            raise ValueError("threshold must be >= 0.")
        self.dt = 1.0 / fps
        self.last_time = time.perf_counter()
        self.threshold = threshold

    def tick(self):
        elapsed = time.perf_counter() - self.last_time
        if elapsed < self.dt:
            time.sleep(self.dt - elapsed)
        # Schedule next tick anchored to the previous schedule,
        # but avoid being more than `threshold` behind the current time.
        self.last_time = max(self.last_time + self.dt, time.perf_counter() - self.threshold)
