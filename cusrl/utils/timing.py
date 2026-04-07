import time
from collections import defaultdict
from contextlib import contextmanager
from typing import DefaultDict

import torch

import cusrl

__all__ = ["Timer", "Rate"]


class _TimerImpl:
    def __init__(self):
        self.start_time = {}
        self.total_time = defaultdict(float)

    def start(self, name):
        raise NotImplementedError

    def stop(self, name):
        raise NotImplementedError

    def get(self, name):
        raise NotImplementedError

    def clear(self):
        self.start_time.clear()
        self.total_time.clear()


class _CpuTimerImpl(_TimerImpl):
    def start(self, name):
        if name in self.start_time:
            raise RuntimeError(f"Timer '{name}' has already been started")
        self.start_time[name] = time.perf_counter()

    def stop(self, name):
        try:
            start_time = self.start_time.pop(name)
        except KeyError as error:
            raise RuntimeError(f"Timer '{name}' has not been started") from error
        self.total_time[name] += time.perf_counter() - start_time

    def get(self, name):
        return self.total_time[name]


class _CudaTimerImpl(_TimerImpl):
    def __init__(self, device: torch.device | str | None = None):
        super().__init__()
        self.device = cusrl.device(device)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA timing requires an available CUDA device")
        if self.device.type != "cuda":
            raise ValueError(f"CUDA timing requires a CUDA device, got '{self.device}'")
        self.pending_time: DefaultDict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)

    def start(self, name):
        if name in self.start_time:
            raise RuntimeError(f"Timer '{name}' has already been started")
        with torch.cuda.device(self.device):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        self.start_time[name] = event

    def stop(self, name):
        try:
            start_event = self.start_time.pop(name)
        except KeyError as error:
            raise RuntimeError(f"Timer '{name}' has not been started") from error
        with torch.cuda.device(self.device):
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
        self.pending_time[name].append((start_event, end_event))

    def get(self, name):
        self._flush(name)
        return self.total_time[name]

    def clear(self):
        super().clear()
        self.pending_time.clear()

    def _flush(self, name: str | None = None):
        names = [name] if name is not None else list(self.pending_time)
        for key in names:
            pending = self.pending_time[key]
            if not pending:
                continue
            for start_event, end_event in pending:
                end_event.synchronize()
                self.total_time[key] += start_event.elapsed_time(end_event) / 1000.0
            pending.clear()


class Timer:
    """Measures and accumulates execution time for named code sections.

    The timer uses CPU wall-clock timing by default. When ``device`` resolves
    to a CUDA device, it measures elapsed GPU time with CUDA events instead.
    Repeated measurements with the same name are accumulated until
    :meth:`clear` is called.

    Args:
        device (torch.device | str | None):
            The device that determines the timing backend. CUDA devices use
            CUDA event timing; all other devices use ``time.perf_counter``.
    """

    impl: _TimerImpl

    def __init__(self, device: torch.device | str | None = None):
        device = cusrl.device(device)
        if device.type == "cuda":
            self.impl = _CudaTimerImpl(device=device)
        else:
            self.impl = _CpuTimerImpl()

    def start(self, name):
        """Starts timing a named section.

        Args:
            name:
                The name of the timed section.

        Raises:
            RuntimeError:
                If the named timer has already been started and not yet
                stopped.
        """
        self.impl.start(name)

    def stop(self, name):
        """Stops timing a named section and accumulates its elapsed time.

        Args:
            name:
                The name of the timed section.

        Raises:
            RuntimeError:
                If the named timer was not started.
        """
        self.impl.stop(name)

    def __getitem__(self, item):
        """Returns the accumulated time for a named section in seconds."""
        return self.impl.get(item)

    def clear(self):
        """Clears all active and accumulated timing data."""
        self.impl.clear()

    def wrap(self, name, func):
        """Wraps a callable so each invocation is timed under ``name``."""

        def wrapper(*args, **kwargs):
            self.start(name)
            result = func(*args, **kwargs)
            self.stop(name)
            return result

        return wrapper

    def decorate(self, name):
        """Returns a decorator that times each call under ``name``."""

        def decorator(func):
            return self.wrap(name, func)

        return decorator

    @contextmanager
    def record(self, name):
        """Context manager that times the enclosed block under ``name``."""
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
            raise ValueError("'fps' must be greater than 0")
        if threshold < 0:
            raise ValueError("'threshold' must be greater than or equal to 0")
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
