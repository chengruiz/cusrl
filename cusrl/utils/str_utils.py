import re
from types import FunctionType
from typing import Any

from cusrl.utils.misc import import_obj

__all__ = [
    "camel_to_snake",
    "format_float",
    "get_class_str",
    "get_function_str",
    "parse_class",
    "parse_function",
]


_REGEX_UPPER_LOWER_SPLIT = re.compile(r"([A-Z]+)([A-Z][a-z])")
_REGEX_LOWER_UPPER_SPLIT = re.compile(r"([a-z\d])([A-Z])")
_REGEX_CLASS_STRING = re.compile(r"<class '([^']+)' from '([^']+)'>")
_REGEX_FUNCTION_STRING = re.compile(r"<function '([^']+)' from '([^']+)'>")


def camel_to_snake(name: str) -> str:
    """Converts a CamelCase string to snake_case."""
    if not name:
        return ""

    s1 = _REGEX_UPPER_LOWER_SPLIT.sub(r"\1_\2", name)
    s2 = _REGEX_LOWER_UPPER_SPLIT.sub(r"\1_\2", s1)
    return s2.lower()


def format_float(number, width):
    """Formats a float to a fixed width string."""
    string = f"{number:.{width}f}"[:width]
    if string[-1] != ".":
        return string
    return " " + string[:-1]


def get_class_str(obj: type | Any) -> str:
    """Returns a string representation of the class of a object."""
    if not isinstance(obj, type):
        obj = type(obj)
    return f"<class '{obj.__qualname__}' from '{obj.__module__}'>"


def get_function_str(obj: FunctionType) -> str:
    """Returns a string representation of a function."""
    if obj.__qualname__ == "<lambda>":
        return repr(obj)
    return f"<function '{obj.__qualname__}' from '{obj.__module__}'>"


def parse_class(name: str) -> type | None:
    """Parses a class from its string representation (e.g.
    "<class 'Class' from 'module'>").

    Args:
        name (str):
            The string representation of the class.

    Returns:
        type | None:
            The parsed class type, or None if failed to parse.
    """
    if match := _REGEX_CLASS_STRING.match(name):
        class_name, module_name = match.groups()
        return import_obj(module_name, class_name)
    return None


def parse_function(name: str) -> type | None:
    """Parses a function from its string representation (e.g.
    "<function 'func' from 'module'>").

    Args:
        name (str):
            The string representation of the function.

    Returns:
        type | None:
            The parsed function object, or None if failed to parse.
    """
    if match := _REGEX_FUNCTION_STRING.match(name):
        class_name, module_name = match.groups()
        return import_obj(module_name, class_name)
    return None
