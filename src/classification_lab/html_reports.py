"""Static HTML with escaped user-controlled labels and no executable model content."""

from html import escape
from .report_contracts import validate_report
import json
from pathlib import Path


def render_html(report):
    if not isinstance(report, dict) or report.get("schema_version") != 1 or "metrics" not in report:
        raise ValueError("Expected a version-one experiment report")
    validate_report(report)
    metrics = report["metrics"]
    title = escape(f"{report['dataset']} / {report['model']}")
    rows = "".join(
        f"<tr><th>{escape(name)}</th><td>{float(metrics[name]):.4f}</td></tr>"
        for name in ("accuracy", "balanced_accuracy", "macro_f1")
    )
    classes = metrics["classes"]
    header = "".join(f"<th>{escape(str(label))}</th>" for label in classes)
    matrix_rows = "".join(
        "<tr><th>"
        + escape(str(label))
        + "</th>"
        + "".join(f"<td>{escape(str(value))}</td>" for value in values)
        + "</tr>"
        for label, values in zip(classes, metrics["confusion_matrix"])
    )
    payload = escape(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    scope = escape(str(report.get("evaluation", "Scope not specified")))
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>body{{font:16px system-ui;max-width:960px;margin:3rem auto;padding:0 1rem;color:#182830}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccd6dd;padding:.6rem;text-align:left}}
pre{{overflow:auto;background:#f0f4f6;padding:1rem}}.scope{{padding:1rem;background:#fff1cc}}</style>
<h1>{title}</h1><p class="scope">{scope}</p><h2>Holdout metrics</h2><table>{rows}</table>
<h2>Confusion matrix</h2><p>Actual rows; predicted columns.</p><table><tr><th>Actual / predicted</th>{header}</tr>{matrix_rows}</table>
<details><summary>Complete report and reproducibility metadata</summary><pre>{payload}</pre></details></html>"""


def write_html(report, path):
    content = render_html(report)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
