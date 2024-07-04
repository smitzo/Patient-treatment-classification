from pathlib import Path
from ..synthetic import GENERATORS, generate


def register(commands):
    command = commands.add_parser("generate", help="Generate explicitly synthetic teaching records")
    command.add_argument("dataset", choices=sorted(GENERATORS))
    command.add_argument("--rows", type=int, default=600)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--output-csv", required=True)
    command.set_defaults(handler=execute)


def execute(args):
    frame = generate(args.dataset, args.rows, args.seed)
    path = Path(args.output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {
        "dataset": args.dataset,
        "kind": "synthetic",
        "rows": len(frame),
        "seed": args.seed,
        "output_csv": str(path),
    }
