import torch

from cusrl.utils.str_utils import (
    camel_to_snake,
    format_float,
    get_class_str,
    get_function_str,
    parse_class,
    parse_function,
    parse_torch_dtype,
)


class ExampleClass:
    pass


def example_function():
    return "ok"


def test_camel_to_snake_handles_empty_acronyms_and_numbers():
    assert camel_to_snake("") == ""
    assert camel_to_snake("HTTPServer2D") == "http_server2_d"
    assert camel_to_snake("PpoAgentFactory") == "ppo_agent_factory"


def test_format_float_keeps_fixed_width_without_trailing_dot():
    assert format_float(1.2345, 4) == "1.23"
    assert format_float(12.3, 3) == " 12"


def test_class_and_function_strings_round_trip():
    assert parse_class(get_class_str(ExampleClass)) is ExampleClass
    assert parse_class(get_class_str(ExampleClass())) is ExampleClass
    assert parse_function(get_function_str(example_function)) is example_function


def test_parse_torch_dtype_accepts_prefixed_and_bare_names():
    assert parse_torch_dtype("float32") is torch.float32
    assert parse_torch_dtype("torch.float64") is torch.float64
