from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from typing import Any, TypeVar

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
    """A circular, step-based storage for time-series or batched data with flexible,
    nested fields. Implements the Mapping[str, NestedTensor] interface to allow intuitive
    indexing, iteration, and membership checks over stored fields.

    Args:
        capacity (int):
            Maximum number of time steps the buffer can hold before wrapping around.
        parallelism (int | None):
            Expected size of the penultimate dimension (e.g. number of agents or batch size).
            If None, this is inferred from the first pushed data.
        device (str | torch.device):
            The torch device on which all tensors will be allocated.

    Attributes:
        capacity (int):
            Current buffer capacity.
        parallelism (int):
            Fixed size of the parallel dimension once set or inferred.
        device (torch.device):
            Device for all internal tensors.
        cursor (int):
            Next write index within [0, capacity).
        full (bool):
            True if the buffer has been filled at least once.
        schema (dict[str, Nested[str]]):
            Describes the nested layout of each named field.
        spec (dict[str, FieldSpec]):
            Metadata for each top-level field (temporal/custom flags).
        storage (dict[str, torch.Tensor]):
            Flat mapping from nested keys to their tensor storage.

    Methods:
        __len__() -> int:
            Number of top-level fields in the buffer.
        __iter__():
            Iterate over top-level field names.
        __contains__(key: str) -> bool:
            Check existence of a top-level field.
        __getitem__(key: str) -> NestedTensor:
            Retrieve a field's data as a nested tensor schema.
        __setitem__(name: str, data: NestedArray) -> None:
            Set or update a top-level field with a nested array-like schema.
        get(key: str, default=None) -> NestedTensor | None:
            Like dict.get, returns default if key is not present.
        clear() -> None:
            Reset buffer state, clearing all data and metadata.
        reset_cursor() -> None:
            Reset the write cursor to zero without clearing stored values.
        resize(capacity: int) -> None:
            Change capacity (clears existing data if changed).
        push(data: dict[str, NestedArray]) -> None:
            Append a new time step; wraps around when capacity is reached.
        add_field(name: str, data: NestedArray, temporal: bool = True) -> None:
            Add or overwrite a custom field (temporal or static).
        sample(sampler: Callable[[str, FieldSpec, torch.Tensor], torch.Tensor]) -> dict[str, NestedTensor]:
            Apply a sampling function to each stored tensor and return a batch with original nesting.

        ValueError:
            On shape mismatches, capacity violations, or schema inconsistencies.
        KeyError:
            When attempting to push data into a custom-flagged field, vice versa.
    """

    FieldSpec = FieldSpec

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
            raise ValueError("Parallelism is not set.")
        return self.parallelism

    def clear(self):
        """Clears all stored data and resetting control variables."""
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
                raise ValueError(f"Capacity mismatch: expected {self.capacity}, got {value.size(0)}.")
            if (storage := self.storage.get(key)) is None:
                # If the field is not custom, it should be temporal
                storage = self._create_storage(value, temporal=True, is_step=False)
                self.storage[key] = storage
            storage.copy_(self._as_tensor(value))

    def __delitem__(self, name: str) -> None:
        # Remove a top-level field and its nested storage
        if name not in self.schema:
            raise KeyError(f"Field '{name}' not found.")
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
        """Adds data of a step to the buffer.

        Parameters:
            data (dict[str, NestedArray]): A dictionary containing named arrays to be added to the buffer.

        Raises:
            ValueError: If the passed data does not match the expected shape ([ [..., ] parallelism, num_channels ]).

        Notes:
            - If the buffer reaches its capacity, it wraps around and starts overwriting from the beginning.
            - The buffer's `full` attribute is set to True when the buffer reaches its capacity for the first time.
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
                raise KeyError(f"Field '{name}' already added by 'add_field'.")
            for key, value in iterate_nested(nested_value, name):
                if (storage := self.storage.get(key)) is None:
                    try:
                        storage = self._create_storage(value)
                        self.storage[key] = storage
                    except ValueError as error:
                        raise ValueError(f"Failed to add field '{key}' shaped '{value.shape}'.") from error
                storage[self.cursor] = self._as_tensor(value)

        self.cursor += 1
        if not self.full and self.cursor == self.capacity:
            self.full = True

    def add_field(self, name: str, data: NestedArray, temporal: bool = True):
        if data is None:
            return

        self._check_data_schema(name, data)
        if (spec := self.spec.get(name)) is None:
            self.spec[name] = FieldSpec(temporal=temporal, custom=True)
        elif spec.temporal != temporal:
            raise ValueError(f"Field '{name}' already added with different temporal setting.")
        elif not spec.custom:
            raise ValueError(f"Field '{name}' already added by 'push'.")

        for key, value in iterate_nested(data, name):
            if (storage := self.storage.get(key)) is None:
                storage = self._create_storage(value, temporal=temporal, is_step=False)
                self.storage[key] = storage
            storage.copy_(self._as_tensor(value))

    def sample(self, sampler: Callable[[str, FieldSpec, torch.Tensor], torch.Tensor]) -> dict[str, NestedTensor]:
        """Samples a batch of data from the storage using the provided sampler function.

        Args:
            sampler (Callable[[str, FieldSpec, torch.Tensor], torch.Tensor]):
                A function that takes a name, a spec and a tensor, and returns a sampled tensor.

        Returns:
            dict[str, NestedTensor]:
                A dictionary mapping string keys to NestedTensor objects, containing the sampled data from the storage.
        """

        batch = {key: sampler(key, self.spec[key.split(".", 1)[0]], self.storage[key]) for key in self.storage}
        return reconstruct_nested(batch, self.schema)

    def _as_tensor(self, data) -> torch.Tensor:
        return torch.as_tensor(data, device=self.device)

    def _create_storage(
        self, data: np.ndarray | torch.Tensor, temporal: bool = True, is_step: bool = True
    ) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = self._as_tensor(data)
        # Each tensor / array should be in shape of [ [..., ] parallelism, num_channels ]
        if len(data.shape) < 2:
            raise ValueError("Shape of arrays must be [ [..., ] parallelism, num_channels ]")
        if self.parallelism is None:
            self.parallelism = data.size(-2)
        elif data.size(-2) != self.parallelism:
            raise ValueError("Shape of arrays must be [ [..., ] parallelism, num_channels ]")
        if is_step:
            # shape: [ capacity, [..., ] parallelism, num_channels ]
            return data.new_zeros(self.capacity, *data.shape)
        if temporal and data.size(0) != self.capacity:
            raise ValueError(f"Capacity mismatch: expected {self.capacity}, got {data.size(0)}.")
        return torch.zeros_like(data)

    def _check_data_schema(self, name: str, data: NestedArray):
        if (schema := self.schema.get(name)) is None:
            self.schema[name] = get_schema(data, name)
        elif schema != (curr_schema := get_schema(data, name)):
            raise ValueError(f"Schema mismatch for field '{name}': expected '{schema}', got '{curr_schema}'.")


class Sampler:
    """Base class for samplers which samples data from a buffer."""

    def __call__(self, buffer: Buffer) -> Iterator[dict[str, NestedTensor | Any]]:
        yield buffer.sample(lambda _1, _2, x: x)
