from collections.abc import Callable, Iterator, Mapping
from typing import Any, TypeVar, overload

from cusrl.utils.misc import MISSING
from cusrl.utils.typing import Nested

__all__ = [
    "flatten_nested",
    "get_schema",
    "iterate_nested",
    "map_nested",
    "reconstruct_nested",
    "zip_nested",
]


_T = TypeVar("_T")
_V = TypeVar("_V")


def concat_key(prefix: Any, separator: Any, suffix: Any) -> str:
    prefix, separator, suffix = str(prefix), str(separator), str(suffix)
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}{separator}{suffix}"


def get_schema(
    value: Nested[_T],
    prefix: str = "",
    max_depth: int | None = None,
    separator: str = ".",
) -> Nested[str]:
    """Generates a schema of path-like strings from a nested structure.

    This function recursively traverses a nested structure containing
    dictionaries, lists, or tuples. It creates a parallel structure where each
    leaf value is replaced by a string representing its "path" from the root.
    Dictionary keys and list/tuple indices are used as path segments, joined by
    dots.

    Args:
        value (Nested[_T]):
            The nested structure (e.g., a dictionary, list, or tuple) to
            process.
        prefix (str, optional):
            The base prefix for building path strings. Mainly for internal
            recursive use. Defaults to "".
        max_depth (int or None, optional):
            The maximum depth to traverse. If None, traverses the entire
            structure. Defaults to None.
        separator (str, optional):
            Separator used to join path segments. Defaults to ".".

    Returns:
        schema (Nested[str]):
            A nested structure with the same structure as the input, where each
            leaf value is a string representing its path.

    Examples:
        >>> get_schema({'a': 1, 'b': {'c': 2}})
        {'a': 'a', 'b': {'c': 'b.c'}}
        >>> get_schema([10, 20, {'key': 30}])
        ['0', '1', {'key': '2.key'}]
    """
    if max_depth is not None:
        if max_depth <= 0:
            return prefix
        max_depth -= 1

    if isinstance(value, Mapping):
        return {
            key: get_schema(
                val,
                concat_key(prefix, separator, key),
                max_depth=max_depth,
                separator=separator,
            )
            for key, val in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            get_schema(
                item,
                concat_key(prefix, separator, i),
                max_depth=max_depth,
                separator=separator,
            )
            for i, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            get_schema(
                item,
                concat_key(prefix, separator, i),
                max_depth=max_depth,
                separator=separator,
            )
            for i, item in enumerate(value)
        ]
    return prefix


@overload
def iterate_nested(
    data: Nested[_T], prefix: str = "", *, max_depth: None = None, separator: str = "."
) -> Iterator[tuple[str, _T]]: ...
@overload
def iterate_nested(
    data: Nested[_T], prefix: str = "", *, max_depth: int | None, separator: str = "."
) -> Iterator[tuple[str, Nested[_T]]]: ...


def iterate_nested(
    data: Nested[_T],
    prefix: str = "",
    *,
    max_depth: int | None = None,
    separator: str = ".",
) -> Iterator[tuple[str, Nested[_T]]]:
    """Generated a flattened view of the nested data.

    This function traverses nested dictionaries, lists, and tuples. It
    generates a flattened view where keys from dictionaries and indices from
    lists/tuples are joined with a separator to form a single path string
    for each leaf value.

    Args:
        data (Nested[_T]):
            The nested structure (dict, list, or tuple) to iterate over.
        prefix (str, optional):
            A prefix to prepend to all generated keys. Mainly for internal
            recursive use. Defaults to "".
        max_depth (int or None, optional):
            The maximum depth to traverse. If None, traverses the entire
            structure. Defaults to None.
        separator (str, optional):
            Separator used to join path segments. Defaults to ".".

    Yields:
        generator (Iterator[tuple[str, _T]]):
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
    if max_depth is not None:
        if max_depth <= 0:
            yield prefix, data
            return
        max_depth -= 1

    if isinstance(data, Mapping):
        for key, value in data.items():
            yield from iterate_nested(
                value,
                concat_key(prefix, separator, key),
                max_depth=max_depth,
                separator=separator,
            )
    elif isinstance(data, (tuple, list)):
        for i, value in enumerate(data):
            yield from iterate_nested(
                value,
                concat_key(prefix, separator, i),
                max_depth=max_depth,
                separator=separator,
            )
    else:
        yield prefix, data


@overload
def flatten_nested(
    data: Nested[_T], prefix: str = "", *, max_depth: None = None, separator: str = "."
) -> dict[str, _T]: ...
@overload
def flatten_nested(
    data: Nested[_T], prefix: str = "", *, max_depth: int | None, separator: str = "."
) -> dict[str, Nested[_T]]: ...


def flatten_nested(
    data: Nested[_T],
    prefix: str = "",
    *,
    max_depth: int | None = None,
    separator: str = ".",
) -> dict[str, _T] | dict[str, Nested[_T]]:
    """Flattens a nested data structure into a flat dictionary.

    Args:
        data (Nested[_T]):
            The nested structure to flatten.
        prefix (str, optional):
            A prefix to be added to all keys in the flattened dictionary.
            Defaults to "".
        max_depth (int or None, optional):
            The maximum depth to traverse. If None, traverses the entire
            structure. Defaults to None.
        separator (str, optional):
            Separator used to join path segments. Defaults to ".".

    Returns:
        flattened_data (dict[str, _T]):
            A new dictionary with flattened key-value pairs. Keys represent the
            path to the value in the original nested structure.

    Example:
        >>> data = {'a': 1, 'b': {'c': 2, 'd': 3}}
        >>> flatten_nested(data)
        {'a': 1, 'b.c': 2, 'b.d': 3}
    """
    return dict(iterate_nested(data, prefix, max_depth=max_depth, separator=separator))


def map_nested(func: Callable[[_T], _V], data: Nested[_T]) -> Nested[_V]:
    """Applies a function to each leaf element of a nested structure.

    This function traverses a nested dictionaries, lists, and tuples, applies
    the provided function `func` to each leaf value, and returns a new nested
    structure of the same structure with the transformed values.

    Args:
        func (Callable[[_T], _V]):
            A function to apply to each leaf value in the nested structure.
        data (Nested[_T]):
            The nested structure to process.

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
def reconstruct_nested(flattened_data: dict[str, _T], schema: str) -> _T: ...
@overload
def reconstruct_nested(flattened_data: dict[str, _T], schema: Mapping[str, Nested[str]]) -> dict[str, Nested[_T]]: ...
@overload
def reconstruct_nested(flattened_data: dict[str, _T], schema: list[Nested[str]]) -> list[Nested[_T]]: ...
@overload
def reconstruct_nested(flattened_data: dict[str, _T], schema: tuple[Nested[str], ...]) -> tuple[Nested[_T], ...]: ...
@overload
def reconstruct_nested(flattened_data: dict[str, _T], schema: Nested[str]) -> Nested[_T]: ...


def reconstruct_nested(flattened_data: dict[str, _T], schema: Nested[str]) -> Nested[_T]:
    """Reconstructs a nested structure from a flat dictionary and a schema.

    This function takes a flat dictionary of key-value pairs and a nested
    structure (the "schema") where the leaves are string keys. It builds a new
    nested structure that mirrors the structure of the schema, but with the leaf
    keys replaced by their corresponding values from the flat `storage`
    dictionary.

    This is the inverse operation of flattening a nested structure.

    Args:
        flattened_data (dict[str, _T]):
            A flat dictionary mapping string keys to values.
        schema (Nested[str]):
            A nested structure (dict, list, or tuple) where the leaves are
            string keys that are present in the `storage` dict.

    Returns:
        reconstructed_data (Nested[_T]):
            A new nested structure with the same structure as `schema`, but with
            the string keys at the leaves replaced by their corresponding values
            from `storage`.

    Example:
        >>> flattened_data = {'a': 10, 'b.c': 20, 'b.d': 30}
        >>> schema = {'a': 'a', 'b': ('c', 'd')}
        >>> reconstruct_nested(flattened_data, schema)
        {'a': 10, 'b': (20, 30)}
    """
    if isinstance(schema, Mapping):
        return {key: reconstruct_nested(flattened_data, name) for key, name in schema.items()}
    if isinstance(schema, tuple):
        return tuple(reconstruct_nested(flattened_data, name) for name in schema)
    if isinstance(schema, list):
        return [reconstruct_nested(flattened_data, name) for name in schema]
    return flattened_data[schema]


def zip_nested(
    *args: Nested[object],
    prefix: str = "",
    max_depth: int | None = None,
    separator: str = ".",
) -> Iterator[tuple[str, tuple[object, ...]]]:
    """Zips multiple nested structures in lock-step.

    This function traverses nested hierarchies in parallel. At each node:
    - If all inputs have the same structure at this level, recurse.
    - Otherwise, stop descending and yield the current path with the tuple of
      the current sub-objects.

    Leaves (non-container values) are yielded as (path, (v1, v2, ...)).

    Args:
        *args:
            One or more nested structures made of dict, list, and tuple.
        prefix:
            A prefix to prepend to all generated paths. Mainly for internal
            recursive use. Defaults to "".
        max_depth:
            Maximum additional levels to descend; None for unlimited.
        separator:
            Separator used to join path segments. Defaults to ".".

    Yields:
        Iterator[tuple[str, tuple[object, ...]]]: Pairs of (path, values), where
        `values` contains the corresponding sub-objects from each input.

    Examples:
        >>> a1 = {"x": [1, 2], "y": {"z": 3}}
        >>> b1 = {"x": [10, 20], "y": {"z": 30}}
        >>> list(zip_nested(a1, b1))
        [('x.0', (1, 10)), ('x.1', (2, 20)), ('y.z', (3, 30))]

        >>> # Missing values are filled with MISSING
        >>> a2 = {"x": [1, 2, 3]}
        >>> b2 = {"x": [10, 20]}
        >>> list(zip_nested(a2, b2))
        [('x.0', (1, 10)), ('x.1', (2, 20)), ('x.2', (3, MISSING))]

        >>> a3 = {"a": 1, "b": 2}
        >>> b3 = {"a": 10, "c": 30}
        >>> list(zip_nested(a3, b3))
        [('a', (1, 10)), ('b', (2, MISSING)), ('c', (MISSING, 30))]

        >>> # Limit traversal depth
        >>> list(zip_nested(a1, b1, max_depth=1))
        [('x', ([1, 2], [10, 20])), ('y', ({'z': 3}, {'z': 30}))]

    """
    if not args:
        return
    if len(args) == 1:
        for prefix, value in iterate_nested(
            args[0],
            prefix=prefix,
            max_depth=max_depth,
            separator=separator,
        ):
            yield prefix, (value,)
        return

    # Handle depth limit
    if max_depth is not None:
        if max_depth <= 0:
            yield prefix, tuple(args)
            return
        max_depth -= 1

    if any(arg is MISSING for arg in args):
        yield prefix, tuple(args)
        return

    flat_args = [flatten_nested(arg, max_depth=1, separator=separator) for arg in args]
    all_keys = []
    for flat_arg in flat_args:
        for key in flat_arg.keys():
            if key not in all_keys:
                all_keys.append(key)

    if len(all_keys) == 1 and all_keys[0] == "":
        # All are leaf nodes at this level, yield them
        yield prefix, tuple(arg[""] for arg in flat_args)
        return

    for key in all_keys:
        yield from zip_nested(
            *[flat_arg.get(key, MISSING) for flat_arg in flat_args],
            prefix=concat_key(prefix, separator, key),
            max_depth=max_depth,
            separator=separator,
        )
