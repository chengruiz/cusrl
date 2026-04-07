"""Evaluate an agent with a registered experiment."""

import argparse
import sys
from collections.abc import Sequence

import cusrl
from cusrl.utils import cli_utils
from cusrl.utils.tyro_utils import cli as tyro_cli

__all__ = ["main"]


PROGRAM_NAME = "python -m cusrl play"


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description=__doc__)
    # fmt: off
    parser.add_argument("-env", "--environment", type=str, metavar="NAME",
                        help="Name of the environment for playing")
    parser.add_argument("-alg", "--algorithm", type=str, metavar="NAME",
                        help="Name of the algorithm used during training")
    parser.add_argument("--checkpoint", type=str, metavar="PATH",
                        help="Path to a checkpoint to play")
    parser.add_argument("--seed", type=int, metavar="N",
                        help="Seed for reproducibility (default: random)")
    parser.add_argument("-m", "--module", nargs=argparse.REMAINDER, default=(), metavar="MODULE [ARG ...]",
                        help="Run library module as a script, with its arguments")
    parser.add_argument("-s", "--script", nargs=argparse.REMAINDER, default=(), metavar="SCRIPT [ARG ...]",
                        help="Script to run, with its arguments")
    # fmt: on
    parser.epilog = "Pass tyro overrides after '--'"
    if argv is None:
        argv = sys.argv[1:]
    args, extra_args = cli_utils.split_double_dash(argv)
    known_args, unknown_args = parser.parse_known_args(args)
    return known_args, unknown_args + extra_args


def main(argv: Sequence[str] | None = None):
    args, extra_args = parse_args(argv)
    cusrl.set_global_seed(args.seed)
    cli_utils.import_module_from_args(args)
    trial = cli_utils.load_checkpoint_from_args(args)
    experiment = cli_utils.load_experiment_spec_from_args(args)
    player_factory = experiment.to_playing_factory()
    prog = f"{PROGRAM_NAME} --environment {args.environment} --algorithm {args.algorithm} --"
    player_factory = tyro_cli(prog=prog, default=player_factory, args=extra_args)
    player_factory(trial).run_playing_loop()


if __name__ == "__main__":
    main()
