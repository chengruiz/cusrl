import inspect
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any, TypeVar, overload

import torch

from cusrl.utils.misc import MISSING
from cusrl.utils.str_utils import get_class_str, get_function_str, parse_class, parse_function

__all__ = [
    "from_dict",
    "get_first",
    "prefix_dict_keys",
    "to_dict",
]


_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")
_D = TypeVar("_D")


def from_dict(obj, data: dict[str, Any] | Any) -> Any:
    """Builds a Python object from a nested mapping / sequence description.

    This function converts primitives and recursively reconstructs lists,
    tuples, and dicts. When an existing object is provided, it performs a
    comparison with it to preserve unchanged attributes / items from
    that object, updating only the parts that differ. A special MISSING sentinel
    indicates that keys or sequence items should be removed.

    Behavior summary:
    - Primitives (int, float, bool, None) pass through unchanged.
    - Strings are returned as-is, unless they conform "<class '*' from '*'>",
      in which case it is recognized as a class.
    - When obj is None, the entire structure is recursively converted.
    - When obj is provided:
        - For each top-level key / index, the function compares the flattened
            forms of the current value and the incoming value. If equal, the
            existing value from obj is kept.
        - Otherwise, the value is recursively rebuilt with the current value.
    - If data is a dict with a "__class__" entry that is a type:
        - slice: returns slice(start, stop, step).
        - torch.device: returns torch.device(str_value).
        - If the type defines `from_dict`, it is used; otherwise the type is
          instantiated with the remaining key-value pairs as keyword arguments.

    Args:
        obj:
            Existing object to update or use as context. If None, data is fully
            reconstructed. May be an object, dict, list, or tuple.
        data:
            The description of the value to build. May be a primitive, string,
            dict, list, or tuple. Dicts may include a "__class__" type entry.

    Returns:
        The reconstructed value or object.

    Raises:
        NotImplementedError: If an unexpected container type is encountered.
        ValueError: If a "__class__" entry is present but is not a type.
    """
    if isinstance(data, (int, float, bool, type(None), type(MISSING))):
        return data
    if isinstance(data, str):
        if cls := parse_class(data):
            return cls
        if func := parse_function(data):
            return func
        return data

    if obj is None or obj is MISSING:
        if isinstance(data, (list, tuple)):
            data = type(data)(from_dict(None, item) for item in data)
        elif isinstance(data, dict):
            data = {key: from_dict(None, value) for key, value in data.items()}
        else:
            raise NotImplementedError(f"Unexpected data type '{type(data)}'.")
    else:
        from cusrl.utils.nest import flatten_nested, zip_nested

        for key, (current_value_dict, updated_value_dict) in zip_nested(to_dict(obj), data, max_depth=1):
            if hasattr(obj, key):
                current_value = getattr(obj, key)
            elif isinstance(obj, dict):
                current_value = obj.get(key)
            elif isinstance(obj, (list, tuple)):
                index = int(key)
                current_value = obj[index] if index < len(obj) else None
            elif current_value_dict is not MISSING:
                current_value = current_value_dict
            else:
                current_value = None

            # Checks for equality
            if flatten_nested(current_value_dict) == flatten_nested(updated_value_dict):
                # Keeps the current value retrieved from the object
                if isinstance(data, dict):
                    data[key] = current_value
                elif isinstance(data, (list, tuple)):
                    data = type(data)([*data[: int(key)], current_value, *data[int(key) + 1 :]])
                else:
                    raise NotImplementedError(f"Unexpected data type '{type(data)}'.")
                continue

            updated_value = from_dict(current_value, updated_value_dict)
            if isinstance(data, dict):
                if updated_value is not MISSING:
                    data[key] = updated_value
                else:
                    data.pop(key, None)
            elif isinstance(data, (list, tuple)):
                if updated_value is not MISSING:
                    data = type(data)([*data[: int(key)], updated_value, *data[int(key) + 1 :]])
                else:
                    data = type(data)([*data[: int(key)], *data[int(key) + 1 :]])
            else:
                raise NotImplementedError(f"Unexpected data type '{type(data)}'.")

    if isinstance(data, dict) and (cls := data.pop("__class__", None)):
        if not isinstance(cls, type):
            raise ValueError(f"Class '{cls}' is not correctly parsed.")
        if cls is slice:
            return slice(data["start"], data["stop"], data["step"])
        if cls is torch.device:
            return torch.device(data["__str__"])
        if hasattr(cls, "from_dict"):
            return cls.from_dict(data)
        return cls(**data)
    return data


@overload
def get_first(data: Mapping[_K, _V], *keys: _K) -> _V: ...
@overload
def get_first(data: Mapping[_K, _V], *keys: _K, default: _V | _D) -> _V | _D: ...


def get_first(data: Mapping[_K, _V], *keys, default: _V | _D = MISSING) -> _V | _D:
    for key in keys:
        if (value := data.get(key, MISSING)) is not MISSING:
            return value
    if default is not MISSING:
        return default
    raise KeyError(str(keys))


def prefix_dict_keys(data: Mapping[str, _T], prefix: str) -> dict[str, _T]:
    """Adds a prefix to all keys in the dictionary."""
    return {f"{prefix}{key}": value for key, value in data.items()}


def to_dict(obj) -> dict[str, Any] | Any:
    """Converts an object to a dictionary representation."""
    if hasattr(obj, "to_dict"):
        obj_dict = obj.to_dict()

    # If the object is not a dictorionary-convertable object
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_dict(item) for item in obj)
    elif isinstance(obj, type):
        return get_class_str(obj)
    elif inspect.isfunction(obj):
        return get_function_str(obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    elif isinstance(obj, slice):
        obj_dict = {"start": obj.start, "stop": obj.stop, "step": obj.step}
    elif is_dataclass(obj):
        obj_dict = {field.name: getattr(obj, field.name) for field in fields(obj)}
    elif isinstance(obj, Mapping):
        obj_dict = dict(obj)
    else:
        obj_dict = {}
        for slot in getattr(obj, "__slots__", ()):
            if hasattr(obj, slot):
                obj_dict[slot] = getattr(obj, slot)
        for key, value in getattr(obj, "__dict__", {}).items():
            if not key.startswith("_"):
                obj_dict[key] = value
        if not obj_dict:
            obj_dict = {"__str__": str(obj)}

    obj_dict = {key: to_dict(value) for key, value in obj_dict.items()}
    if not isinstance(obj, dict):
        obj_dict = {"__class__": get_class_str(obj)} | obj_dict
    return obj_dict
