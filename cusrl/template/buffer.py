from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from typing import Any, TypeAlias, TypeVar, overload

import torch

import cusrl
from cusrl.utils.nest import get_schema, iterate_nested, reconstruct_nested
from cusrl.utils.typing import Nested, NestedArray, NestedTensor

__all__ = ["Buffer", "Sampler"]


_T = TypeVar("_T")


class Buffer(MutableMapping[str, NestedTensor]):
    """Circular storage for nested tensors keyed by top-level field name.

    `push()` appends one time step per field. Each leaf is expected to have
    shape `[parallelism, ...]`; the shared `parallelism` dimension is
    either provided up front or inferred from the first value written. Temporal
    fields are stored with a leading capacity axis and wrap around once the
    buffer is full.

    The mapping interface operates on top-level field names. Nested leaves are
    flattened internally into `storage`. Assigning through the mapping
    interface replaces an entire field and expects each leaf to already have
    shape `[capacity, parallelism, ...]`.
    """

    def __init__(
        self,
        capacity: int,
        parallelism: int,
        device: str | torch.device | None = None,
    ):
        self.capacity: int = capacity
        self.parallelism: int = parallelism
        self.device = cusrl.device(device)

        self.cursor = 0
        self.full = False

        self.schema: dict[str, Nested[str]] = {}
        self.storage: dict[str, torch.Tensor] = {}

    def get_parallelism(self) -> int:
        return self.parallelism

    def clear(self):
        """Clears all stored data and resets control variables."""
        self.cursor = 0
        self.full = False
        self.storage.clear()
        self.schema.clear()

    def reset_cursor(self):
        """Resets the buffer's step counter to zero."""
        self.cursor = 0

    def resize(self, capacity: int):
        if capacity == self.capacity:
            return
        self.clear()
        self.capacity = capacity

    def __iter__(self):
        yield from self.schema

    def __contains__(self, key):
        return key in self.schema

    def __getitem__(self, key):
        return reconstruct_nested(self.storage, self.schema[key])

    def __setitem__(self, name, data: NestedArray):
        """Register or overwrite a full field value.

        Args:
            name (str):
                Top-level field name.
            data (NestedArray):
                Nested array-like payload for the field. Each leaf must have
                shape `[capacity, parallelism, ...]`.

        Raises:
            ValueError:
                If the schema changes or the field shape is incompatible with
                the buffer capacity or parallelism.
        """
        if data is None:
            return

        self._check_data_schema(name, data)
        for key, value in iterate_nested(data, name):
            value = self._as_tensor(value)
            self._validate_field_shape(key, value.shape)
            if (storage := self.storage.get(key)) is None:
                storage = torch.zeros_like(value, device=self.device)
                self.storage[key] = storage
            storage.copy_(value)

    def __delitem__(self, name: str) -> None:
        if name not in self.schema:
            raise KeyError(f"Field '{name}' was not found")
        for _, key in iterate_nested(self.schema[name]):
            del self.storage[key]
        del self.schema[name]

    def __len__(self):
        return len(self.schema)

    @overload
    def get(self, key: str, default: None = None) -> NestedTensor | None: ...

    @overload
    def get(self, key: str, default: _T) -> NestedTensor | _T: ...

    def get(self, key: str, default: NestedTensor | _T | None = None) -> NestedTensor | _T | None:
        if (struct := self.schema.get(key)) is None:
            return default
        return reconstruct_nested(self.storage, struct)

    def push(self, data: Mapping[str, NestedArray]):
        """Append one step for each temporal field in `data`.

        Each leaf must have shape `[parallelism, ...]`. The first write for a
        field fixes its nested schema and allocates storage of shape
        `[capacity, parallelism, ...]`. Once `capacity` steps have been written,
        the buffer is marked full and the cursor wraps back to index ``0``.

        Raises:
            ValueError:
                If a leaf shape is incompatible or the nested schema changes.
        """
        for name, nested_value in data.items():
            if nested_value is None:
                continue
            self._check_data_schema(name, nested_value)
            for key, value in iterate_nested(nested_value, name):
                value = self._as_tensor(value)
                if (storage := self.storage.get(key)) is None:
                    self._validate_step_shape(key, value.shape)
                    storage = value.new_zeros(self.capacity, *value.shape)
                    self.storage[key] = storage
                storage[self.cursor] = value

        self.cursor += 1
        if self.cursor == self.capacity:
            self.full = True
            self.cursor = 0

    def sample(self, sampler: Callable[[str, torch.Tensor], torch.Tensor]) -> dict[str, NestedTensor]:
        """Apply `sampler` to every stored leaf and rebuild the nested result.

        The callback receives the flattened leaf name and the backing storage
        tensor. Its return value replaces the stored tensor in the sampled
        batch.
        """

        batch = {key: sampler(key, self.storage[key]) for key in self.storage}
        return reconstruct_nested(batch, self.schema)

    def _as_tensor(self, data) -> torch.Tensor:
        return torch.as_tensor(data, device=self.device)

    def _validate_step_shape(self, name: str, shape: Sequence[int]):
        if len(shape) < 2:
            raise ValueError(f"A step of field '{name}' must have shape [parallelism, ...]")

        if shape[0] != self.parallelism:
            raise ValueError(f"Parallelism mismatch for field '{name}': expected {self.parallelism}, got {shape[0]}")

    def _validate_field_shape(self, name: str, shape: Sequence[int]):
        if len(shape) < 3:
            raise ValueError(f"Field '{name}' must have shape [capacity, parallelism, ...]")

        if shape[0] != self.capacity:
            raise ValueError(f"Capacity mismatch for field '{name}': expected {self.capacity}, got {shape[0]}")
        if shape[1] != self.parallelism:
            raise ValueError(f"Parallelism mismatch for field '{name}': expected {self.parallelism}, got {shape[1]}")

    def _check_data_schema(self, name: str, data: NestedArray):
        if (schema := self.schema.get(name)) is None:
            self.schema[name] = get_schema(data, name)
        elif schema != (curr_schema := get_schema(data, name)):
            raise ValueError(f"Schema mismatch for field '{name}': expected '{schema}', got '{curr_schema}'")


BatchMetaData: TypeAlias = dict[str, Any]


class Sampler:
    """Base class for iterators that yield samples from a `Buffer`."""

    def __call__(self, buffer: Buffer) -> Iterator[tuple[BatchMetaData, dict[str, NestedTensor]]]:
        """Yield metadata and sampled batches from ``buffer``.

        Args:
            buffer (Buffer):
                Source buffer whose stored fields will be sampled.

        Yields:
            tuple[BatchMetaData, dict[str, NestedTensor]]:
                Pairs of per-batch metadata and sampled nested tensors.
        """
        yield {}, buffer.sample(lambda _name, tensor: tensor)
