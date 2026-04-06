"""List available experiments"""

import argparse
import sys
from collections.abc import Sequence

from cusrl.utils import cli_utils
from cusrl.zoo import load_experiment_modules, registry

__all__ = ["parse_args", "main"]

PROGRAM_NAME = "python -m cusrl list-experiments"


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description=__doc__)

    # fmt: off
    parser.add_argument("-m", "--module", nargs=argparse.REMAINDER, metavar="MODULE [ARG ...]",
                        help="Run library module as a script, with its arguments")
    parser.add_argument("script", nargs=argparse.REMAINDER, metavar="SCRIPT [ARG ...]",
                        help="Script to run, with its arguments")
    # fmt: on
    if argv is None:
        argv = sys.argv[1:]
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    cli_utils.import_module_from_args(args)
    load_experiment_modules()
    print("Available experiments:", end="")
    print("".join([f"\n- {experiment_name}" for experiment_name in sorted(registry.keys())]))


if __name__ == "__main__":
    main()
