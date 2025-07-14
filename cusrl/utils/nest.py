from collections.abc import Callable, Iterator, Mapping
from typing import TypeVar, overload

from cusrl.utils.typing import Nested

__all__ = [
    "get_schema",
    "iterate_nested",
    "map_nested",
    "reconstruct_nested",
    "zip_nested",
]


_T = TypeVar("_T")


def get_schema(value: Nested[_T], prefix: str = "") -> Nested[str]:
    if isinstance(value, Mapping):
        if prefix:
            prefix += "."
        return {key: get_schema(val, f"{prefix}{key}") for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        if prefix:
            prefix += "."
        return tuple(get_schema(item, f"{prefix}{i}") for i, item in enumerate(value))
    return prefix


def iterate_nested(data: Nested[_T], prefix: str = "") -> Iterator[tuple[str, _T]]:
    if isinstance(data, Mapping):
        if prefix:
            prefix += "."
        for key, value in data.items():
            yield from iterate_nested(value, f"{prefix}{key}")
    elif isinstance(data, (tuple, list)):
        if prefix:
            prefix += "."
        for i, value in enumerate(data):
            yield from iterate_nested(value, f"{prefix}{i}")
    else:
        yield prefix, data


def map_nested(data: Nested[_T], func: Callable[[_T], _T]) -> Nested[_T]:
    structure = get_schema(data)
    result = {}
    for key, value in iterate_nested(data):
        result[key] = func(value)
    return reconstruct_nested(result, structure)


@overload
def reconstruct_nested(storage: dict[str, _T], schema: str) -> _T: ...
@overload
def reconstruct_nested(storage: dict[str, _T], schema: Mapping[str, Nested[str]]) -> dict[str, Nested[_T]]: ...
@overload
def reconstruct_nested(
    storage: dict[str, _T], schema: tuple[Nested[str], ...] | list[Nested[str]]
) -> tuple[_T, ...]: ...


def reconstruct_nested(storage: dict[str, _T], schema: Nested[str]) -> Nested[_T]:
    if isinstance(schema, Mapping):
        return {key: reconstruct_nested(storage, name) for key, name in schema.items()}
    if isinstance(schema, (tuple, list)):
        return tuple(reconstruct_nested(storage, name) for name in schema)
    return storage[schema]


def zip_nested(*args: Nested[_T]) -> Iterator[tuple[str, tuple[_T, ...]]]:
    if not args:
        return

    flat_args = [dict(iterate_nested(arg)) for arg in args]
    keys = sorted(flat_args[0].keys())

    if not all(sorted(flat_arg.keys()) == keys for flat_arg in flat_args[1:]):
        raise ValueError("All nested structures must have the same schema.")

    for path in keys:
        yield path, tuple(flat_arg[path] for flat_arg in flat_args)
