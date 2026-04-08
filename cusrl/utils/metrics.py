import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = ["Metrics"]


@dataclass(slots=True)
class Metric:
    mean: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    count: int = 0

    @torch.no_grad()
    def update(self, mean: torch.Tensor, count: int):
        if count == 0:
            return

        if self.count == 0:
            self.mean = mean.clone()
            self.count = count
            return
        mean = mean.to(self.mean.device)
        total_count = self.count + count
        self.mean.mul_(self.count / total_count).add_(mean * (count / total_count))
        self.count = total_count


class Metrics:
    def __init__(self):
        self._data: dict[str, Metric] = {}

    def clear(self):
        self._data.clear()

    def __getitem__(self, name: str) -> Metric:
        return self._data[name]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def get(self, name: str, default=None):
        return self._data.get(name, default)

    @torch.no_grad()
    def record(self, metrics: Mapping[str, Any | None] | None = None, /, **kwargs: Any | None):
        """Records statistics for multiple metrics.

        Each keyword argument represents a metric name and its corresponding
        value, which can be converted to a tensor via torch.as_tensor.

        Args:
            **kwargs:
                Metric names mapped to values convertible to torch tensors.
        """
        for name, value in itertools.chain((metrics or {}).items(), kwargs.items()):
            if value is None:
                continue
            try:
                value = torch.as_tensor(value, dtype=torch.float32)
            except Exception as error:
                raise ValueError(f"Failed to update metric '{name}'") from error
            numel = value.numel()
            if numel == 0:
                continue
            self._data.setdefault(name, Metric()).update(value.mean(), numel)

    def summary(self, prefix: str = "") -> dict[str, float]:
        """Generates summary statistics with optional prefix.

        Args:
            prefix (str, optional):
                The prefix for all metric names.

        Returns:
            metrics (dict[str, float]):
                A dictionary containing the mean values of all recorded metrics,
                with keys prefixed by the specified prefix.
        """
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return {f"{prefix}{name}": metric.mean.item() for name, metric in self.items()}
