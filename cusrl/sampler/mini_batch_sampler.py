import functools
from collections.abc import Iterator, Sequence
from typing import Any

import torch

from cusrl.template import Buffer, Sampler
from cusrl.utils.typing import NestedTensor

__all__ = ["AutoMiniBatchSampler", "MiniBatchSampler", "TemporalMiniBatchSampler"]


class MiniBatchSampler(Sampler):
    def __init__(self, num_epochs: int = 1, num_mini_batches: int | Sequence[int] = 1, shuffle: bool = True):
        self.num_epochs = num_epochs
        self.num_mini_batches = tuple(
            [num_mini_batches] * self.num_epochs if isinstance(num_mini_batches, int) else num_mini_batches
        )
        if len(self.num_mini_batches) != self.num_epochs:
            raise ValueError(
                "'num_mini_batches' must be a single integer or a sequence of integers with length "
                f"equal to 'num_epochs' ({self.num_epochs}), but got {len(self.num_mini_batches)}."
            )

        self.shuffle = shuffle

    def __call__(self, buffer: Buffer) -> Iterator[dict[str, NestedTensor | Any]]:
        if not buffer.full:
            raise RuntimeError("MiniBatchSampler requires a full buffer to sample from.")
        num_samples = self._get_num_samples(buffer)
        epoch_indices = torch.randperm(num_samples, device=buffer.device)
        for epoch in range(self.num_epochs):
            num_mini_batches = self.num_mini_batches[epoch]
            mini_batch_size = num_samples // num_mini_batches
            if self.shuffle and epoch > 0:
                torch.randperm(num_samples, device=buffer.device, out=epoch_indices)
            for mini_batch_idx in range(num_mini_batches):
                indices = epoch_indices[mini_batch_idx * mini_batch_size : (mini_batch_idx + 1) * mini_batch_size]
                mini_batch: dict[str, Any] = buffer.sample(functools.partial(self._sample, indices=indices))
                mini_batch["epoch_index"] = epoch
                mini_batch["mini_batch_index"] = mini_batch_idx
                mini_batch["total_epochs"] = self.num_epochs
                mini_batch["total_mini_batches"] = num_mini_batches
                yield mini_batch

    def _get_num_samples(self, buffer: Buffer) -> int:
        """Returns the total number of samples in the buffer."""
        return buffer.capacity * buffer.get_parallelism()

    def _sample(self, name: str, field_info: Buffer.FieldSpec, data, indices):
        """Samples data from the buffer based on the provided indices."""
        return data.flatten(0, -2)[indices]


class TemporalMiniBatchSampler(MiniBatchSampler):
    def _get_num_samples(self, buffer: Buffer) -> int:
        return buffer.get_parallelism()

    def _sample(self, name: str, field_info: Buffer.FieldSpec, data, indices):
        result = data[..., indices, :]
        if name.split(".")[0].endswith("memory") and field_info.temporal:
            result = result[0, ...]
        return result


class AutoMiniBatchSampler(Sampler):
    def __init__(self, num_epochs: int = 1, num_mini_batches: int | Sequence[int] = 1, shuffle: bool = True):
        self.num_epochs = num_epochs
        self.num_mini_batches = num_mini_batches
        self.shuffle = shuffle

    def __call__(self, buffer: Buffer) -> Iterator[dict[str, NestedTensor | Any]]:
        is_temporal = any(key.split(".")[0].endswith("memory") for key in buffer)
        sampler_cls = TemporalMiniBatchSampler if is_temporal else MiniBatchSampler
        sampler = sampler_cls(self.num_epochs, self.num_mini_batches, self.shuffle)
        return sampler(buffer)
