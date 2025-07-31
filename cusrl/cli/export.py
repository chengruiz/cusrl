import argparse
import os

from cusrl.cli import utils as cli_utils

__all__ = ["configure_parser", "main"]


def configure_parser(parser):
    # fmt: off
    parser.add_argument("-env", "--environment", type=str, metavar="NAME",
                        help="Name of the environment used during training")
    parser.add_argument("-alg", "--algorithm", type=str, metavar="NAME",
                        help="Name of the algorithm used during training")
    parser.add_argument("--checkpoint", type=str, metavar="PATH",
                        help="Path to a checkpoint to export")
    parser.add_argument("--output-dir", type=str, default="exported", metavar="DIR",
                        help="Directory to save exported files to (default: exported)")
    parser.add_argument("--format", type=str, choices=["onnx", "jit"], default="onnx",
                        help="Target format for export (default: onnx)")
    parser.add_argument("--silent", action="store_true",
                        help="Whether to suppress output messages")
    parser.add_argument("--dynamo", action="store_true",
                        help="Whether to use PyTorch Dynamo for onnx export")
    parser.add_argument("--load-experiment-spec", action="store_true",
                        help="Whether to load experiment spec from checkpoint directory")
    parser.add_argument("--environment-args", type=str, metavar="ARG",
                        help="Additional arguments for the environment")
    parser.add_argument("-m", "--module", nargs=argparse.REMAINDER, metavar="MODULE [ARG ...]",
                        help="Run library module as a script, with its arguments")
    parser.add_argument("script", nargs=argparse.REMAINDER, metavar="SCRIPT [ARG ...]",
                        help="Script to run, with its arguments")
    # fmt: on


def main(args):
    cli_utils.import_module_from_args(args)
    trial = cli_utils.load_checkpoint_from_args(args)
    experiment = cli_utils.load_experiment_spec_from_args(args, trial)
    environment = experiment.make_playing_env(cli_utils.process_environment_args(args))
    agent_factory = experiment.make_agent_factory()
    agent = agent_factory.from_environment(environment)
    if trial is not None:
        checkpoint = trial.load_checkpoint(map_location=agent.device)
        agent.load_state_dict(checkpoint["agent"])
        environment.load_state_dict(checkpoint["environment"])
    agent.export(
        os.path.join(args.output_dir, experiment.name, getattr(trial, "name", "dummy")),
        dynamo=args.dynamo,
        target_format=args.format,
        verbose=not args.silent,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export an agent for deployment")
    configure_parser(parser)
    main(parser.parse_args())
