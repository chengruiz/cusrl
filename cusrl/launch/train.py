import argparse

import cusrl
from cusrl.utils import cli as cli_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an agent in an environment using a specified algorithm.")
    # fmt: off
    parser.add_argument("-env", "--environment", type=str, required=True, metavar="NAME",
                        help="Name of the environment for training")
    parser.add_argument("-alg", "--algorithm", type=str, required=True, metavar="NAME",
                        help="Name of the algorithm to use")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["none", "wandb", "tensorboard"],
                        help="Logger for statistics tracking (default: tensorboard)")
    parser.add_argument("--log-dir", type=str, default="logs", metavar="DIR",
                        help="Directory to save logs to (default: logs)")
    parser.add_argument("--name", type=str,
                        help="Name of this trial")
    parser.add_argument("--log-interval", type=int, default=1, metavar="N",
                        help="Interval to log at (default: 1)")
    parser.add_argument("--seed", type=int, metavar="N",
                        help="Seed for reproducibility (default: random)")
    parser.add_argument("--num-iterations", type=int, metavar="N",
                        help="Override the number of iterations to train for")
    parser.add_argument("--init-iteration", type=int, metavar="N",
                        help="Initial iteration number")
    parser.add_argument("--save-interval", type=int, metavar="N",
                        help="Override the interval to save checkpoints at")
    parser.add_argument("--checkpoint", type=str, metavar="PATH",
                        help="Path to a checkpoint to resume training from")
    parser.add_argument("--device", type=str,
                        help="Device to use for training")
    parser.add_argument("--autocast", action="store_true",
                        help="Whether to use automatic mixed precision")
    parser.add_argument("--compile", action="store_true",
                        help="Whether to use `torch.compile`")
    parser.add_argument("--environment-args", type=str, metavar="ARG",
                        help="Additional arguments for the environment")
    parser.add_argument("-m", "--module", nargs=argparse.REMAINDER, metavar="MODULE [ARG ...]",
                        help="Run library module as a script, with its arguments")
    parser.add_argument("script", nargs=argparse.REMAINDER, metavar="SCRIPT [ARG ...]",
                        help="Script to run, with its arguments")
    # fmt: on
    args = parser.parse_args()

    cusrl.set_global_seed(args.seed)
    cli_utils.import_module_from_args(args)
    experiment = cusrl.zoo.get_experiment(args.environment, args.algorithm)
    experiment.make_trainer(
        environment_kwargs=cli_utils.process_environment_args(args),
        agent_factory_kwargs={"device": args.device, "autocast": args.autocast, "compile": args.compile},
        logger_factory=cusrl.logger.make_factory(
            args.logger,
            log_dir=f"{args.log_dir}/{experiment.name}",
            name=args.name,
            interval=args.log_interval,
        ),
        num_iterations=args.num_iterations,
        init_iteration=args.init_iteration,
        save_interval=args.save_interval,
        checkpoint_path=args.checkpoint,
    ).run_training_loop()
