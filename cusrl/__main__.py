import argparse

from cusrl.cli import benchmark, export, find_trial, list_experiments, play, train


def main():
    parser = argparse.ArgumentParser(prog="python -m cusrl")
    subparsers = parser.add_subparsers()

    parser_benchmark = subparsers.add_parser(
        "benchmark",
        help="Benchmark an agent with a registered experiment",
        add_help=False,
    )
    parser_benchmark.set_defaults(func=benchmark.main)

    parser_export = subparsers.add_parser(
        "export",
        help="Export an agent for deployment",
        add_help=False,
    )
    parser_export.set_defaults(func=export.main)

    parser_find_trial = subparsers.add_parser(
        "find-trial",
        help="Find a trial directory or checkpoint path",
        add_help=False,
    )
    parser_find_trial.set_defaults(func=find_trial.main)

    parser_list_exp = subparsers.add_parser(
        "list-experiments",
        help="List available experiments",
        add_help=False,
    )
    parser_list_exp.set_defaults(func=list_experiments.main)

    parser_play = subparsers.add_parser(
        "play",
        help="Evaluate an agent with a registered experiment",
        add_help=False,
    )
    parser_play.set_defaults(func=play.main)

    parser_train = subparsers.add_parser(
        "train",
        help="Train an agent with a registered experiment",
        add_help=False,
    )
    parser_train.set_defaults(func=train.main)

    args, extra_args = parser.parse_known_args()
    if hasattr(args, "func"):
        args.func(extra_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
