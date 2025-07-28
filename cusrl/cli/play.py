import argparse

import cusrl
from cusrl.cli import utils as cli_utils

__all__ = ["configure_parser", "main"]


def configure_parser(parser):
    # fmt: off
    parser.add_argument("-env", "--environment", type=str, metavar="NAME",
                        help="Name of the environment for playing")
    parser.add_argument("-alg", "--algorithm", type=str, metavar="NAME",
                        help="Name of the algorithm to use")
    parser.add_argument("--checkpoint", type=str, metavar="PATH",
                        help="Path to the checkpoint file or directory")
    parser.add_argument("--load-experiment-spec", action='store_true',
                        help="Whether to load experiment spec from the checkpoint directory")
    parser.add_argument("--num-steps", type=int, metavar="N",
                        help="Number of steps to run the player for (default: infinite)")
    parser.add_argument("--timestep", type=float, metavar="T",
                        help="Override the timestep of the environment")
    parser.add_argument("--seed", type=int, metavar="N",
                        help="Random seed (default: random)")
    parser.add_argument("--stochastic", action='store_true',
                        help="Whether to use stochastic actions instead of deterministic")
    parser.add_argument("--environment-args", type=str, metavar="ARG",
                        help="Additional arguments for the environment")
    parser.add_argument("-m", "--module", nargs=argparse.REMAINDER, metavar="MODULE [ARG ...]",
                        help="Run library module as a script, with its arguments")
    parser.add_argument("script", nargs=argparse.REMAINDER, metavar="SCRIPT [ARG ...]",
                        help="Script to run, with its arguments")
    # fmt: on


def main(args):
    cusrl.set_global_seed(args.seed)
    cli_utils.import_module_from_args(args)
    trial = cli_utils.load_checkpoint_from_args(args)
    experiment = cli_utils.load_experiment_spec_from_args(args, trial)
    experiment.make_player(
        environment_kwargs=cli_utils.process_environment_args(args),
        checkpoint_path=trial,
        num_steps=args.num_steps,
        timestep=args.timestep,
        deterministic=not args.stochastic,
        verbose=True,
    ).run_playing_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an agent with a registered experiment")
    configure_parser(parser)
    main(parser.parse_args())
