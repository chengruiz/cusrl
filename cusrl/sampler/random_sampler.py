import functools
from typing import Any

import torch

from cusrl.template import Buffer, Sampler

__all__ = ["AutoRandomSampler", "RandomSampler", "TemporalRandomSampler"]


class RandomSampler(Sampler):
    def __init__(self, num_batches: int, batch_size: int):
        self.num_batches = num_batches
        self.batch_size = batch_size

    def __call__(self, buffer: Buffer):
        num_samples = self._get_num_samples(buffer)
        for batch_index in range(self.num_batches):
            metadata = {
                "batch_index": batch_index,
                "total_batches": self.num_batches,
            } | self._get_metadata()
            indices = torch.randint(num_samples, (self.batch_size,), device=buffer.device)
            mini_batch: dict[str, Any] = buffer.sample(functools.partial(self._sample, indices=indices))
            yield metadata, mini_batch

    def _get_metadata(self) -> dict[str, Any]:
        return {"temporal": False}

    def _get_num_samples(self, buffer: Buffer) -> int:
        """Returns the total number of samples in the buffer."""
        return (buffer.capacity if buffer.full else buffer.cursor) * buffer.get_parallelism()

    def _sample(self, name: str, data: torch.Tensor, indices):
        """Samples data from the buffer based on the provided indices."""
        return data.movedim(0, -3).flatten(-3, -2)[indices]


class TemporalRandomSampler(RandomSampler):
    def _get_metadata(self) -> dict[str, Any]:
        return {"temporal": True}

    def _get_num_samples(self, buffer: Buffer) -> int:
        return buffer.get_parallelism()

    def _sample(self, name: str, data: torch.Tensor, indices):
        result = data[..., indices, :]
        if name.split(".")[0].endswith("memory"):
            result = result[0, ...]
        return result


class AutoRandomSampler(Sampler):
    def __init__(self, num_batches: int, batch_size: int):
        self.num_batches = num_batches
        self.batch_size = batch_size

    def __call__(self, buffer: Buffer):
        is_temporal = any(key.split(".")[0].endswith("memory") for key in buffer)
        sampler_cls = TemporalRandomSampler if is_temporal else RandomSampler
        sampler = sampler_cls(num_batches=self.num_batches, batch_size=self.batch_size)
        return sampler(buffer)
