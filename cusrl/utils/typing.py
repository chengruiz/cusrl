from collections.abc import Mapping, Sequence
from typing import TypeAlias, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

__all__ = [
    "Action",
    "Array",
    "ArrayType",
    "BoolArrayType",
    "Done",
    "Info",
    "ListOrTuple",
    "Memory",
    "Nested",
    "NestedArray",
    "NestedTensor",
    "Observation",
    "Reward",
    "Slice",
    "State",
    "StateType",
    "Terminated",
    "Truncated",
]

Array: TypeAlias = np.ndarray | torch.Tensor
Slice: TypeAlias = slice | Sequence[int]
ArrayType = TypeVar("ArrayType", np.ndarray, torch.Tensor)

_T = TypeVar("_T")
ListOrTuple: TypeAlias = list[_T] | tuple[_T, ...]
Nested: TypeAlias = _T | ListOrTuple["Nested[_T]"] | Mapping[str, "Nested[_T]"]
NestedArray = Nested[np.ndarray] | Nested[torch.Tensor]
NestedTensor = Nested[torch.Tensor]

Observation: TypeAlias = Array
Action: TypeAlias = Array
State: TypeAlias = Array | None
StateType = TypeVar("StateType", np.ndarray, torch.Tensor, None)
Reward: TypeAlias = Array
Done: TypeAlias = NDArray[np.bool_] | torch.Tensor
Terminated: TypeAlias = NDArray[np.bool_] | torch.Tensor
Truncated: TypeAlias = NDArray[np.bool_] | torch.Tensor
BoolArrayType = TypeVar("BoolArrayType", NDArray[np.bool_], torch.Tensor)
Info: TypeAlias = dict[str, Nested[Array]]
ValidMemory: TypeAlias = torch.Tensor | tuple["ValidMemory", ...]
Memory: TypeAlias = ValidMemory | None
