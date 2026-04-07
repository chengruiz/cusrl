"""Find a cusrl trial directory or checkpoint path."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import cusrl

__all__ = ["parse_args", "main"]

PROGRAM_NAME = "python -m cusrl find-trial"


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description=__doc__)
    # fmt: off
    parser.add_argument("-env", "--environment", required=True, metavar="NAME",
                        help="Environment name used for training")
    parser.add_argument("-alg", "--algorithm", required=True, metavar="NAME",
                        help="Algorithm name used for training")
    parser.add_argument("--log-dir", type=str, default="logs", metavar="DIR",
                        help="Root logs directory (default: logs)")
    parser.add_argument("--name", type=str, default=None, metavar="NAME",
                        help="Substring to filter trial directory names by")
    parser.add_argument("--basename", action="store_true",
                        help="Print only the basename instead of the full path")
    parser.add_argument("--list", action="store_true",
                        help="Whether to list all matching trial directories instead of printing the latest one")
    parser.add_argument("--ckpt", action="store_true",
                        help="Whether to print latest checkpoint path under the selected trial directory")
    # fmt: on
    if argv is None:
        argv = sys.argv[1:]
    return parser.parse_args(argv)


def print_trial_path(trial: Path, *, print_ckpt: bool = False, print_basename: bool = False):
    if print_ckpt:
        try:
            trial = cusrl.Trial(trial, verbose=False).checkpoint_path
        except ValueError:
            pass
    print(trial.name if print_basename else str(trial))


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    experiment_homes = [
        (Path(args.log_dir) / f"{args.environment}_{args.algorithm}").absolute(),
        (Path(args.log_dir) / f"{args.environment}:{args.algorithm}").absolute(),
    ]
    experiment_homes = list(dict.fromkeys(experiment_homes))
    existing_experiment_homes = [path for path in experiment_homes if path.exists()]
    if not existing_experiment_homes:
        attempted = " or ".join(f"'{path}'" for path in experiment_homes)
        raise FileNotFoundError(f"No experiment directory was found at {attempted}")

    # Find trial directories under the matching experiment homes
    trial_dirs: list[Path] = [
        path
        for experiment_home in existing_experiment_homes
        for path in experiment_home.iterdir()
        if path.is_dir() and not path.is_symlink() and (path / "ckpt").exists()
    ]
    trial_dirs.sort(key=lambda path: path.name, reverse=True)
    if args.name:
        trial_dirs = [path for path in trial_dirs if args.name in path.name]
    if not trial_dirs:
        name_hint = f" with name containing '{args.name}'" if args.name else ""
        homes = ", ".join(f"'{path}'" for path in existing_experiment_homes)
        raise FileNotFoundError(f"No trial directories were found under {homes}{name_hint}")

    if args.list:
        for path in trial_dirs:
            print_trial_path(path, print_ckpt=args.ckpt, print_basename=args.basename)
        return

    print_trial_path(trial_dirs[0], print_ckpt=args.ckpt, print_basename=args.basename)


if __name__ == "__main__":
    main()
