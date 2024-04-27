"""Command-line entrypoint with machine-readable stdout."""

import argparse
import json
from .datasets import load_dataset
from .experiments import train
from .reporting import write_json


def parser():
    root = argparse.ArgumentParser(prog="classification-lab")
    commands = root.add_subparsers(dest="command", required=True)
    command = commands.add_parser("train", help="Train a model on a stratified holdout")
    command.add_argument("dataset")
    command.add_argument("--model", choices=["logistic", "forest", "dummy"], default="logistic")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--test-size", type=float, default=0.25)
    command.add_argument("--data-path", help="Patient CSV path")
    command.add_argument("--output")
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        kwargs = {"path": args.data_path} if args.data_path else {}
        result = train(load_dataset(args.dataset, **kwargs), args.model, args.seed, args.test_size)
        if args.output:
            write_json(args.output, result.report)
        print(json.dumps(result.report, allow_nan=False))
        return 0
    except (ValueError, OSError) as error:
        root.error(str(error))
