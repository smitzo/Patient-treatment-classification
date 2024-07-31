from pathlib import Path
from ..artifacts import load_artifact
from ..csv_io import read_checked_csv
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
    dtype = {column: str for column in artifact.spec.categorical}
    if artifact.spec.text:
        dtype[artifact.spec.text] = str
    frame = read_checked_csv(args.csv, dtype=dtype)
    predictions = predict_frame(artifact, frame)
    path = Path(args.output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    return {
        "rows": len(predictions),
        "classes": artifact.metadata["classes"],
        "output_csv": str(path),
    }
