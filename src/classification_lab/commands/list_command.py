from ..catalog import dataset_catalog
from ..datasets import LOADERS, load_dataset


def register(commands):
    command = commands.add_parser("datasets", help="List offline dataset contracts")
    command.add_argument("--output")
    command.set_defaults(handler=execute)


def execute(args):
    result = []
    for name in sorted(LOADERS):
        if name == "patient":
            from ..datasets import PATIENT

            spec, rows = PATIENT, None
        else:
            data = load_dataset(name)
            spec, rows = data.spec, len(data.frame)
        metadata = next(item for item in dataset_catalog()["datasets"] if item["name"] == name)
        result.append(
            {
                **metadata,
                "name": name,
                "kind": spec.kind,
                "features": list(spec.features),
                "target": spec.target,
                "source": spec.source,
                "rows": rows,
            }
        )
    return {"datasets": result}
