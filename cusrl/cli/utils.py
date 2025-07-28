import argparse
import pickle
from typing import Optional

import cusrl
from cusrl.utils.helper import import_module

__all__ = [
    "import_module_from_args",
    "load_checkpoint_from_args",
    "load_experiment_spec_from_args",
    "process_environment_args",
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


def load_checkpoint_from_args(args: argparse.Namespace):
    if args.checkpoint is not None:
        trial = cusrl.Trial(args.checkpoint)
        if args.environment is None:
            args.environment = trial.environment_name
        if args.algorithm is None:
            args.algorithm = trial.algorithm_name
    else:
        trial = None
    return trial


def load_experiment_spec_from_args(args: argparse.Namespace, trial: Optional["cusrl.Trial"] = None):
    if args.load_experiment_spec:
        if trial is None:
            raise ValueError("A checkpoint path should be provided if '--load-experiment-spec' is specified.")
        spec_path = trial.home / "info" / "experiment_spec.pkl"
        with open(spec_path, "rb") as f:
            experiment = pickle.load(f)
    else:
        if args.environment is None:
            raise ValueError("'--environment' should be specified if it cannot be inferred from the checkpoint path.")
        if args.algorithm is None:
            raise ValueError("'--algorithm' should be specified if it cannot be inferred from the checkpoint path.")
        experiment = cusrl.zoo.get_experiment(args.environment, args.algorithm)
    return experiment


def process_environment_args(args: argparse.Namespace):
    if args.environment_args is None:
        return None
    return {"argv": args.environment_args.strip().split()}
