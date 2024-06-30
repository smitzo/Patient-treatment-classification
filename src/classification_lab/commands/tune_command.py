from ..artifacts import save_artifact
from ..datasets import load_dataset
from ..search import tune


def register(commands):
    command = commands.add_parser("tune", help="Select parameters inside training folds")
    command.add_argument("dataset")
    command.add_argument("--model", choices=["logistic", "forest"], default="logistic")
    command.add_argument("--folds", type=int, default=3)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--data-path")
    command.add_argument("--output")
    command.add_argument("--artifact")
    command.set_defaults(handler=execute)


def execute(args):
    kwargs = {"path": args.data_path} if args.data_path else {}
    data = load_dataset(args.dataset, **kwargs)
    result = tune(data, args.model, args.folds, args.seed)
    if args.artifact:
        save_artifact(result, data.spec, args.artifact)
    return result.report
