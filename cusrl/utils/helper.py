import importlib
import os
import random
import re
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from typing import Any, TypeVar, overload

import numpy as np
import torch

from cusrl.utils import CONFIG, distributed
from cusrl.utils.typing import ListOrTuple

__all__ = [
    "MISSING",
    "camel_to_snake",
    "format_float",
    "from_dict",
    "get_or",
    "import_module",
    "import_obj",
    "prefix_dict_keys",
    "set_global_seed",
    "to_dict",
]


_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")
_D = TypeVar("_D")


class _MISSING_TYPE:
    def __repr__(self):
        return "MISSING"


MISSING = _MISSING_TYPE()


def camel_to_snake(name: str) -> str:
    """Converts a CamelCase string to snake_case."""
    if not name:
        return ""

    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s2 = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def format_float(number, width):
    string = f"{number:.{width}f}"[:width]
    if string[-1] != ".":
        return string
    return " " + string[:-1]


def from_dict(obj, data: dict[str, Any] | Any) -> Any:
    if isinstance(data, (int, float, bool, type(None), _MISSING_TYPE)):
        return data
    if isinstance(data, str):
        if cls := parse_class(data):
            return cls
        return data

    if obj is None:
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
                current_value = obj.get(key, None)
            elif isinstance(obj, (list, tuple)):
                index = int(key)
                current_value = obj[index] if index < len(obj) else None
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
def get_or(data: Mapping[_K, _V], *keys: _K) -> _V: ...
@overload
def get_or(data: Mapping[_K, _V], *keys: _K, default: _V | _D) -> _V | _D: ...


def get_or(data: Mapping[_K, _V], *keys, default: _V | _D = MISSING) -> _V | _D:
    for key in keys:
        if (value := data.get(key, MISSING)) is not MISSING:
            return value
    if default is not MISSING:
        return default
    raise KeyError(str(keys))


def get_type_str(obj: type | Any) -> str:
    """Returns a string representation of the type of the object."""
    if not isinstance(obj, type):
        obj = type(obj)
    return f"<class '{obj.__qualname__}' from '{obj.__module__}'>"


def import_module(
    module_name: str | None = None,
    package: str | None = None,
    path: str | None = None,
    args: ListOrTuple[str] | None = None,
):
    """Imports a Python module by name or from a file path dynamically,
    optionally passing arguments.

    Args:
        module_name (str | None, optional):
            The name of the module to import. Cannot be specified together with
            `path`.
        package (str | None, optional):
            The package name to use as the anchor for relative imports (used
            with `module_name`).
        path (str | None, optional):
            The file path to the module to import. Cannot be specified together
            with `module_name`.
        args (ListOrTuple[str] | None, optional):
            Arguments to pass as `sys.argv` to the module during import.

    Returns:
        module:
            The imported module object, or `None` if neither `module_name` nor
            `path` is specified.

    Raises:
        ValueError:
            If both `module_name` and `path` are specified.
        ImportError:
            If the specified module cannot be found or loaded.
        FileNotFoundError:
            If the specified file path does not exist.
    """

    if module_name and path:
        raise ValueError("'module_name' and 'path' cannot be both specified.")

    if module_name is not None:
        # Check if module is already imported to avoid re-import conflicts
        if module_name in sys.modules:
            return sys.modules[module_name]

        module_spec = find_spec(module_name, package=package)
        if module_spec is None:
            raise ImportError(f"Module '{module_name}' not found.")
    elif path is not None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path '{path}' does not exist.")
        module_name = os.path.basename(path).removesuffix(".py")
        if module_name in sys.modules:
            return sys.modules[module_name]

        module_spec = spec_from_file_location(module_name, path)
        if module_spec is None:
            raise ImportError(f"Module '{path}' not found.")
    else:  # do nothing if no module is specified
        return None

    module = module_from_spec(module_spec)
    sys.modules[module_spec.name] = module

    if module_spec.loader is None:
        return module  # namespace package

    original_argv = sys.argv.copy()
    try:
        sys.argv[:] = [module_spec.origin or "", *(args or [])]
        module_spec.loader.exec_module(module)
    finally:
        sys.argv[:] = original_argv

    return module


def import_obj(module_name: str, qual_name: str) -> Any:
    module = importlib.import_module(module_name)
    if module is None:
        raise ImportError(f"Module '{module_name}' not found.")
    cls = module
    for part in qual_name.split("."):
        cls = getattr(cls, part, None)
    if cls is None:
        raise ImportError(f"'{qual_name}' not found in module '{module_name}'.")
    return cls


def parse_class(name: str) -> type | None:
    """Parses a class from its string representation (e.g.
    "<class 'module.Class'>").

    Args:
        name (str):
            The string representation of the class.

    Returns:
        type | None:
            The parsed class type, or None if the string is not a class.
    """
    if match := re.match(r"<class '([^']+)' from '([^']+)'>", name):
        class_name, module_name = match.groups()
        return import_obj(module_name, class_name)
    return None


def prefix_dict_keys(data: Mapping[str, _T], prefix: str) -> dict[str, _T]:
    """Adds a prefix to all keys in the dictionary."""
    return {f"{prefix}{key}": value for key, value in data.items()}


def set_global_seed(seed: int | None, deterministic: bool = False) -> int:
    """Sets the global random seed for reproducibility.

    Modified from isaacsim.core.utils.set_seed.

    Args:
        seed (int | None):
            The seed to use. If None, a seed will be generated.
        deterministic (bool):
            Whether to use deterministic algorithms.

    Returns:
        seed (int):
            The seed that was set.
    """
    if seed is None:
        seed = 42 if deterministic else int.from_bytes(os.urandom(4), "big")

    if distributed.is_main_process():
        print(f"Setting seed: {seed} (deterministic={deterministic})")
    seed += distributed.local_rank()
    random.seed(seed)
    np.random.seed(random.getrandbits(4))
    torch.manual_seed(random.getrandbits(4))
    os.environ["PYTHONHASHSEED"] = str(random.getrandbits(4))
    torch.cuda.manual_seed(random.getrandbits(4))

    if deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    CONFIG.seed = seed
    return seed


def to_dict(obj) -> dict[str, Any] | Any:
    """Converts an object to a dictionary representation."""
    if hasattr(obj, "to_dict"):
        obj_dict = obj.to_dict()

    # If the object is not a dictorionary-convertable object
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_dict(item) for item in obj)
    elif isinstance(obj, type):
        return get_type_str(obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    elif isinstance(obj, slice):
        obj_dict = {"start": obj.start, "stop": obj.stop, "step": obj.step}
    elif is_dataclass(obj):
        obj_dict = {f.name: getattr(obj, f.name) for f in fields(obj)}
    elif isinstance(obj, Mapping):
        obj_dict = dict(**obj)
    elif hasattr(obj, "__dict__") and obj.__dict__:
        obj_dict = {key: value for key, value in obj.__dict__.items() if not key.startswith("_")}
    else:
        obj_dict = {"__str__": str(obj)}

    obj_dict = {key: to_dict(value) for key, value in obj_dict.items()}
    if not isinstance(obj, dict):
        obj_dict = {"__class__": get_type_str(obj)} | obj_dict
    return obj_dict
