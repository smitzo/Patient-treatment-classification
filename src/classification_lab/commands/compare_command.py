from ..comparison import compare
from ..datasets import load_dataset


def register(commands):
    command = commands.add_parser("compare", help="Compare models using training-selection CV")
    command.add_argument("dataset")
    command.add_argument("--models", nargs="+", default=["dummy", "logistic", "forest"])
    command.add_argument("--folds", type=int, default=3)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--data-path")
    command.add_argument("--output")
    command.set_defaults(handler=execute)


def execute(args):
    kwargs = {"path": args.data_path} if args.data_path else {}
    return compare(load_dataset(args.dataset, **kwargs), args.models, args.folds, args.seed)
