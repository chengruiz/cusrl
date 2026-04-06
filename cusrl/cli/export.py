"""Export an agent for deployment."""

import argparse
import sys
from collections.abc import Sequence

import cusrl
from cusrl.template.agent import Agent
from cusrl.utils import cli_utils
from cusrl.utils.tyro_utils import cli as tyro_cli

__all__ = ["parse_args", "main"]

PROGRAM_NAME = "python -m cusrl export"


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description=__doc__)
    # fmt: off
    parser.add_argument("-env", "--environment", type=str, metavar="NAME",
                        help="Name of the environment used during training")
    parser.add_argument("-alg", "--algorithm", type=str, metavar="NAME",
                        help="Name of the algorithm used during training")
    parser.add_argument("--checkpoint", type=str, metavar="PATH",
                        help="Path to a checkpoint to export")
    parser.add_argument("--output-dir", type=str, metavar="DIR",
                        help="Directory to save exported files to")
    parser.add_argument("--format", type=str, choices=["onnx", "jit"], default="onnx",
                        help="Target format for export (default: onnx)")
    parser.add_argument("--no-optimize", action="store_false", dest="optimize",
                        help="Whether to disable optimization during export")
    parser.add_argument("--silent", action="store_true",
                        help="Whether to suppress output messages")
    parser.add_argument("--batch-size", type=int, default=1, metavar="N",
                        help="Batch size for the exported model")
    parser.add_argument("--opset-version", type=int, default=None, metavar="N",
                        help="ONNX opset version to use for export")
    parser.add_argument("--dynamo", action="store_true",
                        help="Whether to use PyTorch Dynamo for onnx export")
    parser.add_argument("-m", "--module", nargs=argparse.REMAINDER, metavar="MODULE [ARG ...]",
                        help="Run library module as a script, with its arguments")
    parser.add_argument("script", nargs=argparse.REMAINDER, metavar="SCRIPT [ARG ...]",
                        help="Script to run, with its arguments")
    # fmt: on
    parser.epilog = "Pass tyro overrides after '--'"
    if argv is None:
        argv = sys.argv[1:]
    args, extra_args = cli_utils.split_double_dash(argv)
    return parser.parse_args(args), extra_args


def main(argv: Sequence[str] | None = None):
    args, extra_args = parse_args(argv)
    cusrl.config.enable_flash_attention(False)
    cli_utils.import_module_from_args(args)
    trial = cli_utils.load_checkpoint_from_args(args)
    experiment = cli_utils.load_experiment_spec_from_args(args)
    trainer_factory = experiment.to_training_factory()
    prog = f"{PROGRAM_NAME} --environment {args.environment} --algorithm {args.algorithm} --"
    trainer_factory = tyro_cli(prog=prog, default=trainer_factory, args=extra_args)
    environment = trainer_factory.make_environment()
    agent: Agent = trainer_factory.agent_factory.from_environment(environment)
    if trial is not None:
        checkpoint = trial.load_checkpoint(map_location=agent.device)
        agent.load_state_dict(checkpoint["agent"])
        environment.load_state_dict(checkpoint["environment"])
        if args.output_dir is None:
            args.output_dir = trial.home / "exported"
    if args.output_dir is None:
        args.output_dir = "exported"
    agent.export(
        output_dir=args.output_dir,
        target_format=args.format,
        optimize=args.optimize,
        batch_size=args.batch_size,
        opset_version=args.opset_version,
        dynamo=args.dynamo,
        verbose=not args.silent,
    )


if __name__ == "__main__":
    main()
