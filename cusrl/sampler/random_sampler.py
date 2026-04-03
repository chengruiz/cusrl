import functools
from dataclasses import dataclass
from typing import Any

import torch

from cusrl.template import Buffer, Sampler

__all__ = ["AutoRandomSampler", "RandomSampler", "TemporalRandomSampler"]


@dataclass(slots=True)
class BufferInfo:
    full: bool
    cursor: int


class RandomSampler(Sampler):
    """Sample independent transitions uniformly from the valid buffer region.

    Args:
        num_batches (int):
            Number of batches to generate per iteration.
        batch_size (int):
            Number of transitions in each sampled batch.
    """

    def __init__(self, num_batches: int, batch_size: int):
        self.num_batches = num_batches
        self.batch_size = batch_size

    def __call__(self, buffer: Buffer):
        num_samples = self._get_num_samples(buffer)
        buffer_info = BufferInfo(full=buffer.full, cursor=buffer.cursor)
        for batch_index in range(self.num_batches):
            metadata = {
                "batch_index": batch_index,
                "total_batches": self.num_batches,
            } | self._get_metadata()
            indices = torch.randint(num_samples, (self.batch_size,), device=buffer.device)
            mini_batch: dict[str, Any] = buffer.sample(
                functools.partial(self._sample, indices=indices, buffer_info=buffer_info)
            )
            yield metadata, mini_batch

    def _get_metadata(self) -> dict[str, Any]:
        return {"temporal": False}

    def _get_num_samples(self, buffer: Buffer) -> int:
        """Returns the total number of samples in the buffer."""
        return (buffer.capacity if buffer.full else buffer.cursor) * buffer.get_parallelism()

    def _sample(self, name: str, data: torch.Tensor, indices, buffer_info: BufferInfo):
        """Samples data from the buffer based on the provided indices."""
        if not buffer_info.full:
            # Ignore the unwritten tail while the circular buffer is still filling
            data = data[: buffer_info.cursor]
        return data.flatten(0, 1)[indices]


class TemporalRandomSampler(Sampler):
    """Sample random temporal windows, each with an independently chosen start
    and environment instance.

    Args:
        num_batches (int):
            Number of batches to generate per iteration.
        batch_size (int):
            Number of temporal sequences in each sampled batch.
        sequence_len (int | None, optional):
            Length of each sampled sequence. If ``None``, the sampler uses the
            full valid temporal extent of the buffer.
    """

    def __init__(self, num_batches: int, batch_size: int, sequence_len: int | None = None):
        if sequence_len is not None and sequence_len <= 0:
            raise ValueError("'sequence_len' must be positive or None")
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.sequence_len = sequence_len

    def __call__(self, buffer: Buffer):
        buffer_info = BufferInfo(full=buffer.full, cursor=buffer.cursor)
        valid_sequence_len = buffer.capacity if buffer.full else buffer.cursor
        sequence_len = valid_sequence_len if self.sequence_len is None else min(self.sequence_len, valid_sequence_len)
        if sequence_len == 0:
            raise RuntimeError("TemporalRandomSampler can sample only from a non-empty buffer")
        # Start positions are sampled in logical time, i.e. after reordering a full
        # circular buffer so that `cursor` becomes the oldest valid step
        num_starts = valid_sequence_len - sequence_len + 1
        for batch_index in range(self.num_batches):
            metadata = {
                "batch_index": batch_index,
                "total_batches": self.num_batches,
                "temporal": True,
            }
            # Each sampled sequence chooses its own environment id and start step
            env_indices = torch.randint(buffer.get_parallelism(), (self.batch_size,), device=buffer.device)
            start_indices = torch.randint(num_starts, (self.batch_size,), device=buffer.device)
            time_indices = start_indices.unsqueeze(0) + torch.arange(sequence_len, device=buffer.device).unsqueeze(1)
            if buffer_info.full:
                # Map logical time back to physical ring-buffer positions
                time_indices = (buffer_info.cursor + time_indices) % buffer.capacity
            mini_batch: dict[str, Any] = buffer.sample(
                functools.partial(self._sample, env_indices=env_indices, time_indices=time_indices)
            )
            yield metadata, mini_batch

    def _sample(self, name: str, data: torch.Tensor, env_indices, time_indices):
        return data[time_indices, env_indices.unsqueeze(0)]


class AutoRandomSampler(Sampler):
    """Dispatch to the temporal or non-temporal random sampler based on buffer
    contents.

    Args:
        num_batches (int):
            Number of batches to generate per iteration.
        batch_size (int):
            Number of samples or sequences in each batch, depending on the
            selected sampler.
        sequence_len (int | None, optional):
            Temporal sequence length forwarded to ``TemporalRandomSampler``. It
            is ignored when the buffer is sampled as independent transitions.
    """

    def __init__(self, num_batches: int, batch_size: int, sequence_len: int | None = None):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.sequence_len = sequence_len

    def __call__(self, buffer: Buffer):
        is_temporal = any(key.split(".")[0].endswith("memory") for key in buffer)
        sampler_cls = TemporalRandomSampler if is_temporal else RandomSampler
        kwargs = {"sequence_len": self.sequence_len} if is_temporal else {}
        sampler = sampler_cls(num_batches=self.num_batches, batch_size=self.batch_size, **kwargs)
        return sampler(buffer)
