"""JSON command output with explicit local file arguments."""

import argparse
import json
from .datasets import load_dataset
from .experiments import train
from .reporting import write_json
from .artifacts import save_artifact
from .commands import predict_command
from .commands import list_command
from .commands import validate_command
from .commands import compare_command
from .commands import tune_command
from .commands import generate_command
from .commands import evaluate_command
from .commands import inspect_command
from .commands import leaderboard_command
from .commands import report_command
from .commands import batch_command


REGISTRARS = [
    predict_command.register,
    list_command.register,
    validate_command.register,
    compare_command.register,
    tune_command.register,
    generate_command.register,
    evaluate_command.register,
    inspect_command.register,
    leaderboard_command.register,
    report_command.register,
    batch_command.register,
]


def train_command(args):
    kwargs = {"path": args.data_path} if args.data_path else {}
    data = load_dataset(args.dataset, **kwargs)
    result = train(data, args.model, args.seed, args.test_size)
    if args.artifact:
        save_artifact(result, data.spec, args.artifact)
    return result.report


def parser():
    root = argparse.ArgumentParser(prog="classification-lab")
    commands = root.add_subparsers(dest="command", required=True)
    command = commands.add_parser("train", help="Fit a pipeline on a stratified holdout")
    command.add_argument("dataset")
    command.add_argument("--model", choices=["logistic", "forest", "dummy"], default="logistic")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--test-size", type=float, default=0.25)
    command.add_argument("--data-path", help="Patient CSV path")
    command.add_argument("--output")
    command.add_argument("--artifact", help="Write an executable pickle artifact and JSON metadata")
    command.set_defaults(handler=train_command)
    for register in REGISTRARS:
        register(commands)
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        report = args.handler(args)
        if getattr(args, "output", None):
            write_json(args.output, report)
        print(json.dumps(report, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError, KeyError) as error:
        root.error(str(error))
