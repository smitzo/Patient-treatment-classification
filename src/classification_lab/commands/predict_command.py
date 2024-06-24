import csv
from pathlib import Path
import pandas as pd
from ..artifacts import load_artifact
from ..prediction import predict_frame


def register(commands):
    command = commands.add_parser("predict", help="Predict from an explicitly trusted pickle")
    command.add_argument("artifact")
    command.add_argument("csv")
    command.add_argument(
        "--trust-artifact", action="store_true", help="Allow executable pickle deserialization"
    )
    command.add_argument("--output-csv", required=True)
    command.set_defaults(handler=execute)


def execute(args):
    artifact = load_artifact(args.artifact, trusted=args.trust_artifact)
    with Path(args.csv).open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate prediction CSV columns")
    frame = pd.read_csv(args.csv)
    predictions = predict_frame(artifact, frame)
    path = Path(args.output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    return {
        "rows": len(predictions),
        "classes": artifact.metadata["classes"],
        "output_csv": str(path),
    }
