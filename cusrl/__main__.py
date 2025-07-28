import argparse

from cusrl.cli import export, list_experiments, play, train


def main():
    parser = argparse.ArgumentParser(prog="python -m cusrl")
    subparsers = parser.add_subparsers()

    parser_list_exp = subparsers.add_parser("list-experiments", help="List available experiments")
    list_experiments.configure_parser(parser_list_exp)
    parser_list_exp.set_defaults(func=list_experiments.main)

    parser_train = subparsers.add_parser("train", help="Train an agent with a registered experiment")
    train.configure_parser(parser_train)
    parser_train.set_defaults(func=train.main)

    parser_play = subparsers.add_parser("play", help="Evaluate an agent with a registered experiment")
    play.configure_parser(parser_play)
    parser_play.set_defaults(func=play.main)

    parser_export = subparsers.add_parser("export", help="Export an agent for deployment")
    export.configure_parser(parser_export)
    parser_export.set_defaults(func=export.main)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
