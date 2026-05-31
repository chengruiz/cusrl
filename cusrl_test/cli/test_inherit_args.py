import pytest

from cusrl.cli import benchmark, export, play


@pytest.mark.parametrize("module", [play, benchmark, export])
def test_inherit_args_defaults_to_true(module):
    args, extra_args = module.parse_args(["--agent.device", "cpu"])

    assert args.inherit_args is True
    assert extra_args == ["--agent.device", "cpu"]


@pytest.mark.parametrize("module", [play, benchmark, export])
def test_inherit_args_false_is_parsed_as_command_arg(module):
    args, extra_args = module.parse_args(["--inherit-args", "False", "--agent.device", "cpu"])

    assert args.inherit_args is False
    assert extra_args == ["--agent.device", "cpu"]
