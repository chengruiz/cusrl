import argparse

from cusrl.cli import utils as cli_utils
from cusrl.zoo import load_experiment_modules, registry

__all__ = ["configure_parser", "main"]


def configure_parser(parser):
    # fmt: off
    parser.add_argument("-m", "--module", nargs=argparse.REMAINDER, metavar="MODULE [ARG ...]",
                        help="Run library module as a script, with its arguments")
    parser.add_argument("script", nargs=argparse.REMAINDER, metavar="SCRIPT [ARG ...]",
                        help="Script to run, with its arguments")
    # fmt: on


def main(args):
    cli_utils.import_module_from_args(args)
    load_experiment_modules()
    print("Available experiments:", end="")
    print("".join([f"\n- {experiment_name}" for experiment_name in sorted(registry.keys())]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List available experiments")
    configure_parser(parser)
    main(parser.parse_args())
