import importlib
import os
import random
import sys
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from typing import Any

import numpy as np
import torch

from cusrl.utils import CONFIG, distributed
from cusrl.utils.typing import ListOrTuple

__all__ = [
    "MISSING",
    "import_module",
    "import_obj",
    "set_global_seed",
]


class _MISSING_TYPE:
    """A sentinel type used to represent a missing value."""

    def __repr__(self):
        return "MISSING"


MISSING = _MISSING_TYPE()
"""A sentinel value to indicate that a value is missing or not provided."""


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


def import_obj(module_name: str, obj_qualname: str) -> Any:
    """Dynamically import an object (e.g., class, function, variable) from a
    specified module.

    Args:
        module_name (str): Name of the module to import.
        obj_qualname (str): The qualified name (__qualname__) of the object.

    Returns:
        Any: The imported object.

    Raises:
        ImportError: If the module or the object is not found.
    """
    module = importlib.import_module(module_name)
    if module is None:
        raise ImportError(f"Module '{module_name}' not found.")
    cls = module
    for part in obj_qualname.split("."):
        cls = getattr(cls, part, None)
    if cls is None:
        raise ImportError(f"'{obj_qualname}' not found in module '{module_name}'.")
    return cls


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

    distributed.print_rank0(f"Setting seed: {seed} (deterministic={deterministic})")
    seed += distributed.rank()
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
