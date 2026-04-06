import argparse
from collections.abc import Sequence

import cusrl
from cusrl.utils.misc import import_module

__all__ = [
    "import_module_from_args",
    "load_checkpoint_from_args",
    "load_experiment_spec_from_args",
]


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


def split_double_dash(args: Sequence[str]) -> tuple[list[str], list[str]]:
    args = list(args)
    if "--" not in args:
        return args, []
    separator_index = args.index("--")
    return args[:separator_index], args[separator_index + 1 :]
