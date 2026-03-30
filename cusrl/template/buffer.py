from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar

import numpy as np
import torch

from cusrl.utils.nest import get_schema, iterate_nested, reconstruct_nested
from cusrl.utils.typing import Nested, NestedArray, NestedTensor

__all__ = ["Buffer", "Sampler"]


@dataclass(slots=True)
class FieldSpec:
    temporal: bool = True
    custom: bool = False


_T = TypeVar("_T")


class Buffer(MutableMapping[str, NestedTensor]):
    """Circular storage for nested tensors keyed by top-level field name.

    `push()` appends one time step per field. Each leaf is expected to have
    shape `[..., parallelism, channels]`; the shared `parallelism` dimension is
    either provided up front or inferred from the first value written. Temporal
    fields are stored with a leading capacity axis and wrap around once the
    buffer is full.

    Fields registered through `add_field()` are marked as custom. Custom fields
    may be temporal, in which case the caller supplies the full
    `[capacity, ..., parallelism, channels]` tensor, or static, in which case
    the data is stored without a capacity axis. `push()` only writes non-custom
    temporal fields.

    The mapping interface operates on top-level field names. Nested leaves are
    flattened internally into `storage` and reconstructed on access and
    sampling.
    """

    FieldSpec: TypeAlias = FieldSpec

    def __init__(self, capacity: int, parallelism: int | None, device: str | torch.device):
        self.capacity = capacity
        self.device = torch.device(device)

        self.parallelism: int | None = parallelism
        self.cursor = 0
        self.full = False

        self.schema: dict[str, Nested[str]] = {}
        self.spec: dict[str, FieldSpec] = {}
        self.storage: dict[str, torch.Tensor] = {}

    def get_parallelism(self) -> int:
        if self.parallelism is None:
            raise ValueError("Buffer parallelism has not been set")
        return self.parallelism

    def clear(self):
        """Clears all stored data and resets control variables."""
        self.parallelism = None
        self.cursor = 0
        self.full = False
        self.storage.clear()
        self.schema.clear()
        self.spec.clear()

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

    def __setitem__(self, name, data):
        if (spec := self.spec.get(name)) is None or spec.custom:
            self.add_field(name, data)
            return
        # Enable to modify the buffer directly
        self._check_data_schema(name, data)
        for key, value in iterate_nested(data, name):
            if value.size(0) != self.capacity:
                raise ValueError(f"Capacity mismatch: expected {self.capacity}, got {value.size(0)}")
            if (storage := self.storage.get(key)) is None:
                # If the field is not custom, it should be temporal
                storage = self._create_storage(value, temporal=True, sequential=True)
                self.storage[key] = storage
            storage.copy_(self._as_tensor(value))

    def __delitem__(self, name: str) -> None:
        # Remove a top-level field and its nested storage
        if name not in self.schema:
            raise KeyError(f"Field '{name}' was not found")
        # Remove nested storage entries
        for _, key in iterate_nested(self.schema[name]):
            del self.storage[key]
        # Remove schema and spec for the field
        del self.schema[name]
        del self.spec[name]

    def __len__(self):
        return len(self.schema)

    def get(self, key: str, default: _T = None) -> NestedTensor | _T:
        if (struct := self.schema.get(key)) is None:
            return default
        return reconstruct_nested(self.storage, struct)

    def push(self, data: dict[str, NestedArray]):
        """Append one step for each temporal field in `data`.

        Each leaf must have shape `[..., parallelism, channels]`. The first
        write for a field fixes its nested schema and allocates storage of shape
        `[capacity, *leaf.shape]`. Once `capacity` steps have been written, the
        next call wraps around to index `0`.

        Raises:
            KeyError:
                If `data` includes a field previously registered via
                `add_field()`.
            ValueError:
                If a leaf shape is incompatible or the nested schema changes.
        """
        if self.cursor == self.capacity:
            self.cursor = 0

        for name, nested_value in data.items():
            if nested_value is None:
                continue
            self._check_data_schema(name, nested_value)
            if (spec := self.spec.get(name)) is None:
                self.spec[name] = FieldSpec(temporal=True, custom=False)
            elif spec.custom:
                raise KeyError(f"Field '{name}' was already added with 'add_field'")
            for key, value in iterate_nested(nested_value, name):
                if (storage := self.storage.get(key)) is None:
                    try:
                        storage = self._create_storage(value)
                        self.storage[key] = storage
                    except ValueError as error:
                        raise ValueError(f"Failed to push field '{key}' with shape {tuple(value.shape)}") from error
                storage[self.cursor] = self._as_tensor(value)

        self.cursor += 1
        if not self.full and self.cursor == self.capacity:
            self.full = True

    def add_field(self, name: str, data: NestedArray, temporal: bool = True):
        """Register or overwrite a custom field.

        Args:
            name (str):
                Top-level field name.
            data (NestedArray):
                Nested array-like payload for the field.
            temporal (bool, optional):
                Whether `data` includes a leading time axis of length
                `capacity`. Static fields are stored without a capacity axis.
                Defaults to ``True``.

        Raises:
            ValueError:
                If the schema changes, the temporal flag conflicts with a
                previous registration, the field does not match the buffer
                capacity, or `name` already belongs to a field populated by
                `push()`.
        """
        if data is None:
            return

        self._check_data_schema(name, data)
        if (spec := self.spec.get(name)) is None:
            self.spec[name] = FieldSpec(temporal=temporal, custom=True)
        elif spec.temporal != temporal:
            raise ValueError(f"Field '{name}' was already added with a different temporal setting")
        elif not spec.custom:
            raise ValueError(f"Field '{name}' was already added by 'push'")

        for key, value in iterate_nested(data, name):
            if (storage := self.storage.get(key)) is None:
                storage = self._create_storage(value, temporal=temporal, sequential=True)
                self.storage[key] = storage
            storage.copy_(self._as_tensor(value))

    def sample(self, sampler: Callable[[str, FieldSpec, torch.Tensor], torch.Tensor]) -> dict[str, NestedTensor]:
        """Apply `sampler` to every stored leaf and rebuild the nested result.

        The callback receives the flattened leaf name, the top-level `FieldSpec`
        for that field, and the backing storage tensor. Its return value
        replaces the stored tensor in the sampled batch.
        """

        batch = {key: sampler(key, self.spec[key.split(".", 1)[0]], self.storage[key]) for key in self.storage}
        return reconstruct_nested(batch, self.schema)

    def _as_tensor(self, data) -> torch.Tensor:
        return torch.as_tensor(data, device=self.device)

    def _create_storage(
        self,
        data: np.ndarray | torch.Tensor,
        temporal: bool = True,
        sequential: bool = False,
    ) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = self._as_tensor(data)
        # Each tensor / array should be in shape of [ [..., ] parallelism, num_channels ]
        if len(data.shape) < 2:
            raise ValueError("Arrays must have shape [..., parallelism, num_channels]")
        if self.parallelism is None:
            self.parallelism = data.size(-2)
        elif data.size(-2) != self.parallelism:
            raise ValueError("Arrays must have shape [..., parallelism, num_channels]")
        if not sequential:
            # shape: [ capacity, [..., ] parallelism, num_channels ]
            return data.new_zeros(self.capacity, *data.shape)
        if temporal and data.size(0) != self.capacity:
            raise ValueError(f"Capacity mismatch: expected {self.capacity}, got {data.size(0)}")
        return torch.zeros_like(data)

    def _check_data_schema(self, name: str, data: NestedArray):
        if (schema := self.schema.get(name)) is None:
            self.schema[name] = get_schema(data, name)
        elif schema != (curr_schema := get_schema(data, name)):
            raise ValueError(f"Schema mismatch for field '{name}': expected '{schema}', got '{curr_schema}'")


class Sampler:
    """Base class for iterators that yield samples from a `Buffer`."""

    def __call__(self, buffer: Buffer) -> Iterator[tuple[dict[str, Any], dict[str, NestedTensor]]]:
        yield {}, buffer.sample(lambda _1, _2, tensor: tensor)
