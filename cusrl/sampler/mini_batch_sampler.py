import functools
from collections.abc import Sequence
from typing import Any

import torch

from cusrl.template import Buffer, Sampler

__all__ = ["AutoMiniBatchSampler", "MiniBatchSampler", "TemporalMiniBatchSampler"]


class MiniBatchSampler(Sampler):
    """Iterate over shuffled mini-batches of individual transitions from a full
    buffer.

    Args:
        num_epochs (int, optional):
            Number of passes over the buffer.
        num_mini_batches (int | Sequence[int], optional):
            Number of mini-batches per epoch. When a sequence is provided, it
            must contain one value per epoch.
        shuffle (bool, optional):
            Whether to reshuffle indices between epochs. Defaults to ``True``.
    """

    def __init__(self, num_epochs: int = 1, num_mini_batches: int | Sequence[int] = 1, shuffle: bool = True):
        self.num_epochs = num_epochs
        if isinstance(num_mini_batches, int):
            self.num_mini_batches = num_mini_batches
        else:
            self.num_mini_batches = tuple(num_mini_batches)
            if len(self.num_mini_batches) != self.num_epochs:
                raise ValueError(
                    "'num_mini_batches' must be an integer or a sequence of integers with length "
                    f"equal to 'num_epochs' ({self.num_epochs}); got {len(self.num_mini_batches)} values"
                )

        self.shuffle = shuffle

    def __call__(self, buffer: Buffer):
        if not (buffer.full and buffer.cursor == 0):
            raise RuntimeError("MiniBatchSampler requires a full buffer with cursor reset to 0")
        num_samples = self._get_num_samples(buffer)
        epoch_indices = torch.randperm(num_samples, device=buffer.device)
        for epoch in range(self.num_epochs):
            num_mini_batches = (
                self.num_mini_batches if isinstance(self.num_mini_batches, int) else self.num_mini_batches[epoch]
            )
            mini_batch_size = num_samples // num_mini_batches
            if self.shuffle and epoch > 0:
                torch.randperm(num_samples, device=buffer.device, out=epoch_indices)
            for mini_batch_idx in range(num_mini_batches):
                metadata = {
                    "epoch_index": epoch,
                    "mini_batch_index": mini_batch_idx,
                    "total_epochs": self.num_epochs,
                    "total_mini_batches": num_mini_batches,
                } | self._get_metadata()
                indices = epoch_indices[mini_batch_idx * mini_batch_size : (mini_batch_idx + 1) * mini_batch_size]
                mini_batch = buffer.sample(functools.partial(self._sample, indices=indices))
                yield metadata, mini_batch

    def _get_metadata(self) -> dict[str, Any]:
        return {"temporal": False}

    def _get_num_samples(self, buffer: Buffer) -> int:
        """Returns the total number of samples in the buffer."""
        return buffer.capacity * buffer.get_parallelism()

    def _sample(self, name: str, data: torch.Tensor, indices):
        """Samples data from the buffer based on the provided indices."""
        return data.flatten(0, 1)[indices]


class TemporalMiniBatchSampler(MiniBatchSampler):
    """Iterate over shuffled mini-batches of full temporal sequences from a full
    buffer.

    Args:
        num_epochs (int, optional):
            Number of passes over the buffer.
        num_mini_batches (int | Sequence[int], optional):
            Number of mini-batches per epoch. When a sequence is provided, it
            must contain one value per epoch.
        shuffle (bool, optional):
            Whether to reshuffle sequence indices between epochs. Defaults to
            ``True``.
    """

    def _get_metadata(self) -> dict[str, Any]:
        return {"temporal": True}

    def _get_num_samples(self, buffer: Buffer) -> int:
        return buffer.get_parallelism()

    def _sample(self, name: str, data: torch.Tensor, indices):
        result = data[:, indices]
        if name.split(".")[0].endswith("memory"):
            result = result[0, ...]
        return result


class AutoMiniBatchSampler(Sampler):
    """Dispatch to the temporal or non-temporal mini-batch sampler based on
    buffer contents.

    Args:
        num_epochs (int, optional):
            Number of passes over the buffer.
        num_mini_batches (int | Sequence[int], optional):
            Number of mini-batches per epoch. When a sequence is provided, it
            must contain one value per epoch.
        shuffle (bool, optional):
            Whether to reshuffle indices between epochs. Defaults to ``True``.
    """

    def __init__(self, num_epochs: int = 1, num_mini_batches: int | Sequence[int] = 1, shuffle: bool = True):
        self.num_epochs = num_epochs
        self.num_mini_batches = num_mini_batches
        self.shuffle = shuffle

    def __call__(self, buffer: Buffer):
        is_temporal = any(key.split(".")[0].endswith("memory") for key in buffer)
        sampler_cls = TemporalMiniBatchSampler if is_temporal else MiniBatchSampler
        sampler = sampler_cls(self.num_epochs, self.num_mini_batches, self.shuffle)
        return sampler(buffer)
