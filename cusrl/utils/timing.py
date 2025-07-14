import time
from collections import defaultdict
from contextlib import contextmanager

__all__ = ["Timer", "Rate"]


class Timer:
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
    def __init__(self, fps: float):
        self.rate = 1.0 / fps
        self.last_time = time.time()

    def tick(self):
        elapsed = time.time() - self.last_time
        if elapsed < self.rate:
            time.sleep(self.rate - elapsed)
        self.last_time += self.rate
