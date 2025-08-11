import re
from typing import Any

from cusrl.utils.misc import import_obj

__all__ = [
    "camel_to_snake",
    "format_float",
    "get_type_str",
    "parse_class",
]


_REGEX_UPPER_LOWER_SPLIT = re.compile(r"([A-Z]+)([A-Z][a-z])")
_REGEX_LOWER_UPPER_SPLIT = re.compile(r"([a-z\d])([A-Z])")
_REGEX_CLASS_STRING = re.compile(r"<class '([^']+)' from '([^']+)'>")


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


def get_type_str(obj: type | Any) -> str:
    """Returns a string representation of the type of the object."""
    if not isinstance(obj, type):
        obj = type(obj)
    return f"<class '{obj.__qualname__}' from '{obj.__module__}'>"


def parse_class(name: str) -> type | None:
    """Parses a class from its string representation (e.g.
    "<class 'Class' from 'module'>").

    Args:
        name (str):
            The string representation of the class.

    Returns:
        type | None:
            The parsed class type, or None if the string is not a class.
    """
    if match := _REGEX_CLASS_STRING.match(name):
        class_name, module_name = match.groups()
        return import_obj(module_name, class_name)
    return None
