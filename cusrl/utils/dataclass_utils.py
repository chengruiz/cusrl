import inspect
from collections.abc import Mapping
from dataclasses import MISSING
from dataclasses import field as dataclass_field
from dataclasses import fields, is_dataclass, make_dataclass
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from cusrl.utils.str_utils import get_class_str, get_function_str

__all__ = [
    "to_dataclass",
    "to_strict_typed_dataclass",
]


def _get_docstring(obj_type: type[Any]) -> str | None:
    docstring = inspect.getdoc(obj_type)
    if not docstring:
        return None

    default_doc = obj_type.__name__
    if is_dataclass(obj_type):
        default_doc += str(inspect.signature(obj_type)).replace(" -> None", "")
    if docstring == default_doc:
        return None
    return docstring


def _make_dataclass(*args, **kwargs):
    parameters = inspect.signature(make_dataclass).parameters
    kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    return make_dataclass(*args, **kwargs)


def to_dataclass(obj):
    if isinstance(obj, type):
        return get_class_str(obj)
    if hasattr(obj, "to_dict"):
        obj_dict = obj.to_dict()
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_dataclass(item) for item in obj)
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

    obj_dict = {key: to_dataclass(value) for key, value in obj_dict.items()}
    obj_hints = get_type_hints(type(obj), include_extras=True)
    anno_dict = {}

    for key, value in obj_dict.items():
        if is_dataclass(value):
            anno_dict[key] = type(value)
            continue
        anno_dict[key] = obj_hints.get(key, type(value))

    try:
        namespace = {"_original_class": type(obj)}
        if docstring := _get_docstring(type(obj)):
            namespace["__doc__"] = docstring
        datacls = _make_dataclass(
            f"_{type(obj).__name__}DataClass",
            anno_dict.items(),
            namespace=namespace,
            module=type(obj).__module__,
        )
        datacls._original_class = type(obj)
        datacls.__annotations__ = anno_dict
        datacls_obj = datacls(**obj_dict)
        datacls_obj._original_obj = obj
        return datacls_obj
    except TypeError:
        return obj_dict


def _is_dataclass_instance(obj: Any) -> bool:
    return is_dataclass(obj) and not isinstance(obj, type)


def _strictify_field_value(value: Any) -> tuple[Any, type[Any] | None]:
    if _is_dataclass_instance(value):
        strict_value = to_strict_typed_dataclass(value)
        return strict_value, type(strict_value)

    if hasattr(value, "to_dict"):
        strict_value = to_dataclass(value)
        if _is_dataclass_instance(strict_value):
            strict_value = to_strict_typed_dataclass(strict_value)
            return strict_value, type(strict_value)

    return value, None


def _get_field_kwargs(field: Any) -> dict[str, Any]:
    field_kwargs = {
        "init": field.init,
        "repr": field.repr,
        "hash": field.hash,
        "compare": field.compare,
        "metadata": field.metadata,
        "kw_only": field.kw_only,
    }
    if field.default is not MISSING:
        field_kwargs["default"] = field.default
    if field.default_factory is not MISSING:
        field_kwargs["default_factory"] = field.default_factory
    return field_kwargs


def to_strict_typed_dataclass(obj):
    """Returns a dataclass instance whose dataclass-valued fields are narrowed
    to the runtime dataclass types of their current values.

    The returned object is an instance of a dynamically generated subclass of
    the original dataclass, so the original class remains unchanged while
    downstream consumers can inspect stricter field annotations.
    """
    if not _is_dataclass_instance(obj):
        return obj

    obj_type = type(obj)
    obj_hints = get_type_hints(obj_type, include_extras=True)
    obj_field_values = {}
    strict_fields = []

    for source_field in fields(obj_type):
        source_type = obj_hints.get(source_field.name, source_field.type)
        value = getattr(obj, source_field.name)
        value, strict_type = _strictify_field_value(value)
        if strict_type is not None:
            field_type = strict_type
            if get_origin(source_type) is Annotated:
                _, *metadata = get_args(source_type)
                field_type = Annotated[(field_type, *metadata)]

            strict_fields.append((source_field.name, field_type, dataclass_field(**_get_field_kwargs(source_field))))
        obj_field_values[source_field.name] = value

    if not strict_fields:
        return obj

    dataclass_kwargs = {}
    dataclass_params = obj_type.__dataclass_params__
    for attr in dir(dataclass_params):
        if not attr.startswith("_") and hasattr(dataclass_params, attr):
            dataclass_kwargs[attr] = getattr(dataclass_params, attr)

    strict_type = _make_dataclass(
        f"_{obj_type.__name__}StrictDataClass",
        strict_fields,
        bases=(obj_type,),
        namespace={
            "_original_class": getattr(obj, "_original_class", obj_type),
            **({"__doc__": docstring} if (docstring := _get_docstring(obj_type)) else {}),
        },
        module=obj_type.__module__,
        **dataclass_kwargs,
    )

    strict_obj = object.__new__(strict_type)
    for strict_field in fields(strict_type):
        object.__setattr__(strict_obj, strict_field.name, obj_field_values[strict_field.name])
    return strict_obj
