from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Any

import torch

__all__ = ["Metrics"]


@dataclass(slots=True)
class Metric:
    mean: torch.Tensor = None
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


class Metrics(defaultdict[str, Metric]):
    def __init__(self):
        super().__init__(Metric)

    def record(self, **kwargs: Any | None):
        """Records statistics for multiple metrics.

        Each keyword argument represents a metric name and its corresponding value,
        which can be converted to a tensor via torch.as_tensor.

        Args:
            **kwargs: Metric names mapped to values convertible to torch tensors.
        """
        self.update(kwargs)

    @torch.inference_mode()
    def update(self, other: Mapping[str, Any], /, **kwargs):
        for name, value in chain(other.items(), kwargs.items()):
            if value is None:
                continue
            try:
                value = torch.as_tensor(value, dtype=torch.float32)
                numel = value.numel()
                if numel == 0:
                    continue
                self[name].update(value.mean(), numel)
            except Exception as error:
                raise ValueError(f"Error updating metric '{name}'.") from error

    def summary(self, prefix: str = "") -> dict[str, float]:
        """Generates summary statistics with optional prefix.

        Args:
            prefix: Optional prefix for metric names.

        Returns:
            Dictionary containing mean values for all recorded metrics.
        """
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return {f"{prefix}{name}": metric.mean.item() for name, metric in self.items()}
