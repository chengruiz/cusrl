import argparse

from cusrl.cli import utils as cli_utils
from cusrl.template import Agent

__all__ = ["configure_parser", "main"]


def configure_parser(parser):
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
    agent: Agent = agent_factory.from_environment(environment)
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
        dynamo=args.dynamo,
        verbose=not args.silent,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export an agent for deployment")
    configure_parser(parser)
    main(parser.parse_args())
