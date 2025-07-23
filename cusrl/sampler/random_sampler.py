import functools
from collections.abc import Iterator
from typing import Any

import torch

from cusrl.template import Buffer, Sampler
from cusrl.utils.typing import NestedTensor

__all__ = ["AutoRandomSampler", "RandomSampler", "TemporalRandomSampler"]


class RandomSampler(Sampler):
    def __init__(self, num_batches: int, batch_size: int):
        self.num_batches = num_batches
        self.batch_size = batch_size

    def __call__(self, buffer: Buffer) -> Iterator[dict[str, NestedTensor | Any]]:
        num_samples = self._get_num_samples(buffer)
        for batch_index in range(self.num_batches):
            indices = torch.randint(num_samples, (self.batch_size,), device=buffer.device)
            mini_batch: dict[str, Any] = buffer.sample(functools.partial(self._sample, indices=indices))
            mini_batch["batch_index"] = batch_index
            mini_batch["total_batches"] = self.num_batches
            yield mini_batch

    def _get_num_samples(self, buffer: Buffer) -> int:
        """Returns the total number of samples in the buffer."""
        return (buffer.capacity if buffer.full else buffer.cursor) * buffer.get_parallelism()

    def _sample(self, name: str, field_info: Buffer.FieldSpec, data, indices):
        """Samples data from the buffer based on the provided indices."""
        return data.flatten(0, -2)[indices]


class TemporalRandomSampler(RandomSampler):
    def _get_num_samples(self, buffer: Buffer) -> int:
        return buffer.capacity if buffer.full else buffer.cursor

    def _sample(self, name: str, field_info: Buffer.FieldSpec, data, indices):
        result = data[..., indices, :]
        if name.split(".")[0].endswith("memory") and field_info.temporal:
            result = result[0, ...]
        return result


class AutoRandomSampler(Sampler):
    def __init__(self, num_batches: int, batch_size: int):
        self.num_batches = num_batches
        self.batch_size = batch_size

    def __call__(self, buffer: Buffer) -> Iterator[dict[str, NestedTensor | Any]]:
        is_temporal = any(key.split(".")[0].endswith("memory") for key in buffer)
        sampler_cls = TemporalRandomSampler if is_temporal else RandomSampler
        sampler = sampler_cls(num_batches=self.num_batches, batch_size=self.batch_size)
        return sampler(buffer)
