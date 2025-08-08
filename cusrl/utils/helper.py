import importlib
import os
import random
import re
import sys
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from typing import Any, TypeVar, overload

import numpy as np
import torch

from cusrl.utils import CONFIG, distributed
from cusrl.utils.nest import flatten_nested, zip_nested
from cusrl.utils.typing import ListOrTuple

__all__ = [
    "camel_to_snake",
    "format_float",
    "from_dict",
    "get_or",
    "import_module",
    "load_type",
    "prefix_dict_keys",
    "set_global_seed",
    "to_dict",
]


_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")
_D = TypeVar("_D")


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
    if hasattr(obj, "from_dict"):
        data = from_dict(to_dict(obj), data)
        data.pop("__class__", None)
        return obj.from_dict(data)
    if isinstance(data, dict) and data.get("__class__") == "<class 'slice'>":
        return slice(data["start"], data["stop"], data["step"])
    if isinstance(data, str) and (match := re.match(r"<class '([^']+)'>", data)):
        return load_type(match.group(1))
    if not isinstance(obj_dict := to_dict(obj), dict):
        return data
    for key, (value1, value2) in zip_nested(obj_dict, data, max_depth=1):
        # Simple but not efficient check for equality
        if flatten_nested(value1) == flatten_nested(value2):
            continue
        if key == "__class__":
            raise ValueError("Type modification is not supported yet.")
        if hasattr(obj, key):
            setattr(obj, key, from_dict(getattr(obj, key), value2))
        elif isinstance(obj, dict):
            obj[key] = from_dict(obj[key], value2)
        elif isinstance(obj, list):
            index = int(key)
            obj[index] = from_dict(obj[index], value2)
        elif isinstance(obj, tuple):
            index = int(key)
            obj = obj[:index] + (from_dict(obj[index], value2),) + obj[index + 1 :]
        else:
            raise AttributeError(f"Object '{type(obj).__name__}' has no attribute '{key}'.")
    return obj


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


def load_type(full_name: str) -> type:
    """Loads a type by its module and class names.

    Args:
        name (str):
            The fully qualified name of the type (e.g., "torch.nn.Linear").

    Returns:
        type:
            The resolved type object.

    Raises:
        ImportError:
            If the specified type cannot be found.
    """
    if not "." in full_name:
        module_name, class_name = "builtins", full_name
    else:
        module_name, class_name = full_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    if module is None:
        raise ImportError(f"Module '{module_name}' not found.")
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Class '{class_name}' not found in module '{module_name}'.")
    return cls


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


def to_dict(obj, includes_class: bool = True) -> dict[str, Any] | Any:
    """Converts an object to a dictionary representation."""
    if hasattr(obj, "to_dict"):
        obj_dict = obj.to_dict()

    # If the object is not a dictorionary-convertable object
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_dict(item, includes_class=includes_class) for item in obj)
    elif isinstance(obj, type):
        return str(obj)
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

    obj_dict = {key: to_dict(value, includes_class=includes_class) for key, value in obj_dict.items()}
    if includes_class and not isinstance(obj, (dict, OrderedDict)):
        obj_dict = {"__class__": str(type(obj))} | obj_dict
    return obj_dict
