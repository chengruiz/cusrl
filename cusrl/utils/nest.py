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
_V = TypeVar("_V")


def get_schema(value: Nested[_T], prefix: str = "") -> Nested[str]:
    """Generates a schema of path-like strings from a nested structure.

    This function recursively traverses a nested structure containing
    dictionaries, lists, or tuples. It creates a parallel structure where
    each leaf value is replaced by a string representing its "path" from
    the root. Dictionary keys and list/tuple indices are used as path
    segments, joined by dots.

    Args:
        value (Nested[_T]):
            The nested structure (e.g., a dictionary, list, or tuple) to
            process.
        prefix (str, optional):
            The base prefix for building path strings. Mainly for internal
            recursive use. Defaults to "".

    Returns:
        Nested[str]:
            A nested structure with the same structure as the input, where
            each leaf value is a string representing its path.

    Examples:
        >>> get_schema({'a': 1, 'b': {'c': 2}})
        {'a': 'a', 'b': {'c': 'b.c'}}
        >>> get_schema([10, 20, {'key': 30}])
        ('0', '1', {'key': '2.key'})
    """
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
    """Generated a flattened view of the nested data.

    This function traverses nested dictionaries, lists, and tuples. It
    generates a flattened view where keys from dictionaries and indices from
    lists/tuples are joined with a dot ('.') to form a single path string
    for each leaf value.

    Args:
        data (Nested[_T]):
            The nested structure (dict, list, or tuple) to iterate over.
        prefix (str, optional):
            A prefix to prepend to all generated keys. Mainly for internal
            recursive use. Defaults to "".

    Yields:
        Iterator[tuple[str, _T]]:
            An iterator that yields tuples, where each tuple contains a dot-
            separated key path and the corresponding leaf value.

    Example:
        >>> data = {
        ...     "a": 1,
        ...     "b": {
        ...         "c": [10, 20],
        ...         "d": 30,
        ...     },
        ...     "e": (40,),
        ... }
        >>> list(iterate_nested(data))
        [('a', 1), ('b.c.0', 10), ('b.c.1', 20), ('b.d', 30), ('e.0', 40)]
    """
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


def map_nested(data: Nested[_T], func: Callable[[_T], _V]) -> Nested[_V]:
    """Applies a function to each leaf element of a nested structure.

    This function traverses a nested dictionaries, lists, and tuples, applies
    the provided function `func` to each leaf value, and returns a new nested
    structure of the same structure with the transformed values.

    Args:
        data (Nested[_T]):
            The nested structure (e.g., a dictionary of dictionaries) to process.
        func (Callable[[_T], _V]):
            A function to apply to each leaf value in the nested structure.

    Returns:
        Nested[_V]:
            A new nested data with the same structure as `data`, but with `func`
            applied to each leaf value.
    """
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
    """Reconstructs a nested structure from a flat dictionary and a schema.

    This function takes a flat dictionary of key-value pairs and a nested
    structure (the "schema") where the leaves are string keys. It builds a new
    nested structure that mirrors the structure of the schema, but with the leaf
    keys replaced by their corresponding values from the flat `storage` dictionary.

    This is the inverse operation of flattening a nested structure.

    Args:
        storage (dict[str, _T]):
            A flat dictionary mapping string keys to values.
        schema (Nested[str]):
            A nested structure (dict, list, or tuple) where the leaves are string
            keys that are present in the `storage` dict.

    Returns:
        Nested[_T]:
            A new nested structure with the same structure as `schema`, but with the
            string keys at the leaves replaced by their corresponding values from
            `storage`.

    Example:
        >>> storage = {'a': 10, 'b.c': 20, 'b.d': 30}
        >>> schema = {'a': 'a', 'b': ('c', 'd')}
        >>> reconstruct_nested(storage, schema)
        {'a': 10, 'b': (20, 30)}
    """
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
