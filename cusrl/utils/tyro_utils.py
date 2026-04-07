from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from typing import Annotated, Any, TypeAlias, get_args

import torch
import tyro
from torch import nn
from tyro.constructors import ConstructorRegistry, PrimitiveConstructorSpec, PrimitiveTypeInfo

from cusrl.utils.dataclass_utils import to_strict_typed_dataclass
from cusrl.utils.str_utils import parse_torch_dtype

__all__ = ["TYRO_REGISTRY", "JsonDict", "cli", "parse_torch_module_type"]


# A dictionary type, but `tyro` will expect a JSON string from the CLI.
JsonDict: TypeAlias = Annotated[
    dict,
    tyro.constructors.PrimitiveConstructorSpec(
        # Number of arguments to consume.
        nargs=1,
        # Argument name in usage messages.
        metavar="JSON",
        # Convert a list of strings to an instance. The length of the list
        # should match `nargs`.
        instance_from_str=lambda args: json.loads(args[0]),
        # Check if an instance is of the expected type. This is only used for
        # helptext formatting in the presence of union types.
        is_instance=lambda instance: isinstance(instance, dict),
        # Convert an instance to a list of strings. This is used for handling
        # default values that are set in Python. The length of the list should
        # match `nargs`.
        str_from_instance=lambda instance: [json.dumps(instance)],
    ),
]


def parse_torch_module_type(name: str, expected_type: type[nn.Module] = nn.Module) -> type[nn.Module]:
    module_type = getattr(
        nn,
        name.removeprefix("torch.nn.").removeprefix("nn."),
        None,
    )
    if not inspect.isclass(module_type) or not issubclass(module_type, expected_type):
        raise ValueError(f"Unsupported torch.nn.Module type '{name}'")
    return module_type


TYRO_REGISTRY = ConstructorRegistry()


@TYRO_REGISTRY.primitive_rule
def _torch_dtype_rule(type_info: PrimitiveTypeInfo) -> PrimitiveConstructorSpec[torch.dtype] | None:
    if type_info.type is not torch.dtype:
        return None
    return PrimitiveConstructorSpec(
        nargs=1,
        metavar="DTYPE",
        instance_from_str=lambda args: parse_torch_dtype(args[0]),
        is_instance=lambda value: isinstance(value, torch.dtype),
        str_from_instance=lambda value: [str(value)],
    )


@TYRO_REGISTRY.primitive_rule
def _torch_module_type_rule(type_info: PrimitiveTypeInfo) -> PrimitiveConstructorSpec[type[nn.Module]] | None:
    if type_info.type_origin is not type:
        return None

    args = get_args(type_info.type)
    if len(args) != 1:
        return None

    expected_type = args[0]
    if not inspect.isclass(expected_type) or not issubclass(expected_type, nn.Module):
        return None

    return PrimitiveConstructorSpec(
        nargs=1,
        metavar="MODULE",
        instance_from_str=lambda args: parse_torch_module_type(args[0], expected_type=expected_type),
        is_instance=lambda value: inspect.isclass(value) and issubclass(value, expected_type),
        str_from_instance=lambda value: [value.__name__],
    )


def cli(
    f=None,
    *,
    prog: None | str = None,
    default=None,
    config: Sequence[tyro.conf._markers.Marker] | None = (tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
    compact_help: bool = True,
    registry: None | ConstructorRegistry = TYRO_REGISTRY,
    **kwargs: Any,
) -> Any:
    if default is not None:
        strict_default = to_strict_typed_dataclass(default)
        if strict_default is not default:
            default = strict_default
            f = type(strict_default)
    return tyro.cli(
        f,
        prog=prog,
        config=config,
        default=default,
        compact_help=compact_help,
        registry=registry,
        **kwargs,
    )
