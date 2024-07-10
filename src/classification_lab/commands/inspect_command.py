import json
from pathlib import Path


def register(commands):
    command = commands.add_parser(
        "inspect", help="Read JSON metadata only; never deserialize the model"
    )
    command.add_argument("artifact")
    command.add_argument("--output")
    command.set_defaults(handler=execute)


def execute(args):
    metadata = json.loads(Path(str(args.artifact) + ".json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("format_version") != 1:
        raise ValueError("Unknown artifact metadata format")
    return {"metadata": metadata, "scope": "unverified JSON sidecar; model not opened"}
