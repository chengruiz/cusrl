import argparse
from pathlib import Path

import cusrl

__all__ = ["configure_parser", "main"]


def configure_parser(parser: argparse.ArgumentParser):
    # fmt: off
    parser.add_argument("query", metavar="QUERY",
                        help="An experiment path or an experiment name under '--log-dir'")
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


def print_trial_path(trial: Path, *, print_ckpt: bool = False, print_basename: bool = False):
    if print_ckpt:
        try:
            trial = cusrl.Trial(trial, verbose=False).checkpoint_path
        except ValueError:
            pass
    print(trial.name if print_basename else str(trial))


def main(args: argparse.Namespace):
    query = Path(args.query)

    # Get the experiment home directory from the query
    if query.exists():
        experiment_home = query.absolute()
    else:
        experiment_home = (Path(args.log_dir) / query).absolute()
        if not experiment_home.exists():
            raise FileNotFoundError(f"Experiment directory not found: '{query}' or '{experiment_home}'.")

    # Find trial directories under the experiment home
    trial_dirs: list[Path] = [
        path
        for path in experiment_home.iterdir()
        if path.is_dir() and not path.is_symlink() and (path / "ckpt").exists()
    ]
    trial_dirs.sort(key=lambda path: path.name, reverse=True)
    if args.name:
        trial_dirs = [path for path in trial_dirs if args.name in path.name]
    if not trial_dirs:
        name_hint = f" with name containing '{args.name}'" if args.name else ""
        raise FileNotFoundError(f"No trial directories found under '{experiment_home}'{name_hint}.")

    if args.list:
        for path in trial_dirs:
            print_trial_path(path, print_ckpt=args.ckpt, print_basename=args.basename)
        return

    print_trial_path(trial_dirs[0], print_ckpt=args.ckpt, print_basename=args.basename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find a cusrl trial directory or checkpoint path")
    configure_parser(parser)
    main(parser.parse_args())
