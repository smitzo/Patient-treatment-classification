import pandas as pd
from ..metrics import classification_metrics


def register(commands):
    command = commands.add_parser(
        "evaluate", help="Score aligned truth and predicted labels in one CSV"
    )
    command.add_argument("csv")
    command.add_argument("--truth", default="truth")
    command.add_argument("--prediction", default="prediction")
    command.add_argument(
        "--classes", nargs="+", required=True, help="Explicit complete class order"
    )
    command.add_argument("--output")
    command.set_defaults(handler=execute)


def execute(args):
    frame = pd.read_csv(args.csv, dtype=str)
    if args.truth not in frame or args.prediction not in frame:
        raise ValueError("CSV must contain the requested truth and prediction columns")
    if frame[[args.truth, args.prediction]].isna().any().any() or len(set(args.classes)) != len(
        args.classes
    ):
        raise ValueError("Labels must be nonmissing and class order must be unique")
    return {
        "rows": len(frame),
        "metrics": classification_metrics(frame[args.truth], frame[args.prediction], args.classes),
    }
