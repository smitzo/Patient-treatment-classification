from ..datasets import load_dataset
from ..quality import validate_bundle


def register(commands):
    command = commands.add_parser("validate", help="Check a dataset without fitting a model")
    command.add_argument("dataset")
    command.add_argument("--data-path")
    command.add_argument("--output")
    command.set_defaults(handler=execute)


def execute(args):
    kwargs = {"path": args.data_path} if args.data_path else {}
    data = load_dataset(args.dataset, **kwargs)
    return {"dataset": data.spec.name, **validate_bundle(data)}
