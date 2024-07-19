import json
from pathlib import Path
from ..html_reports import write_html
from ..plots import confusion_plot


def register(commands):
    command = commands.add_parser("report", help="Render static HTML from an experiment JSON")
    command.add_argument("json")
    command.add_argument("--html", required=True)
    command.add_argument("--confusion-plot")
    command.add_argument("--normalize", action="store_true")
    command.set_defaults(handler=execute)


def execute(args):
    report = json.loads(Path(args.json).read_text(encoding="utf-8"))
    output = {"html": str(write_html(report, args.html))}
    if args.confusion_plot:
        output["confusion_plot"] = str(
            confusion_plot(report["metrics"], args.confusion_plot, args.normalize)
        )
    return output
