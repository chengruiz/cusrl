import argparse
from collections.abc import Sequence
from typing import Any

import cusrl
from cusrl.utils.misc import import_module

__all__ = [
    "apply_inherited_tyro_args",
    "import_module_from_args",
    "load_checkpoint_from_args",
    "load_experiment_spec_from_args",
    "parse_bool",
]


_INHERITED_TYRO_ARG_PREFIXES = ("--env.", "--agent.")
_INHERITED_TYRO_ARGS = {"--env_args", "--env_kwargs", "--env-args", "--env-kwargs"}
_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def import_module_from_args(args: argparse.Namespace):
    module = args.module
    script = args.script

    if module or script:
        import_module(
            module_name=module[0] if module else None,
            path=script[0] if script else None,
            args=(module or script)[1:],
        )


def load_checkpoint_from_args(args: argparse.Namespace) -> cusrl.Trial | None:
    if args.checkpoint is not None:
        trial = cusrl.Trial(args.checkpoint)
        if args.environment is None:
            args.environment = trial.environment_name
        if args.algorithm is None:
            args.algorithm = trial.algorithm_name
    else:
        trial = None
    return trial


def load_experiment_spec_from_args(args: argparse.Namespace) -> cusrl.zoo.ExperimentSpec:
    if args.environment is None:
        raise ValueError("Specify '--environment' when it cannot be inferred from the checkpoint path")
    if args.algorithm is None:
        raise ValueError("Specify '--algorithm' when it cannot be inferred from the checkpoint path")
    experiment = cusrl.zoo.get_experiment(args.environment, args.algorithm)
    return experiment


def apply_inherited_tyro_args(
    trial: cusrl.Trial | None,
    args: argparse.Namespace,
    extra_args: Sequence[str],
) -> list[str]:
    if not getattr(args, "inherit_args", False) or trial is None:
        return list(extra_args)
    tyro_args = trial.metadata.get("tyro_args", [])
    if not isinstance(tyro_args, list) or not tyro_args:
        return list(extra_args)
    return _filter_inheritable_tyro_args(tyro_args) + list(extra_args)


def _filter_inheritable_tyro_args(tyro_args: Sequence[Any]) -> list[str]:
    inherited_args = []
    consume_next_value = False
    inherit_values_until_option = False
    for arg in tyro_args:
        if not isinstance(arg, str):
            consume_next_value = False
            inherit_values_until_option = False
            continue
        if consume_next_value:
            inherited_args.append(arg)
            consume_next_value = False
            continue
        if arg.startswith("--"):
            has_inline_value = "=" in arg
            is_named_inherited_arg = arg in _INHERITED_TYRO_ARGS
            inheritable = arg.startswith(_INHERITED_TYRO_ARG_PREFIXES) or is_named_inherited_arg
            inherit_values_until_option = inheritable and not is_named_inherited_arg and not has_inline_value
            consume_next_value = is_named_inherited_arg and not has_inline_value
            if inheritable:
                inherited_args.append(arg)
            continue
        if inherit_values_until_option:
            inherited_args.append(arg)
    return inherited_args


def split_double_dash(args: Sequence[str]) -> tuple[list[str], list[str]]:
    args = list(args)
    if "--" not in args:
        return args, []
    separator_index = args.index("--")
    return args[:separator_index], args[separator_index + 1 :]
