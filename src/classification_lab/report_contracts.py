"""Reject malformed report inputs before rendering or ranking them."""

import math
import numbers
import numpy as np


def validate_report(report):
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or report["schema_version"] != 1
    ):
        raise ValueError("Expected a version-one experiment report")
    if any(not isinstance(report.get(key), str) or not report[key] for key in ("dataset", "model")):
        raise ValueError("Report needs dataset and model names")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Report needs classification metrics")
    classes = metrics.get("classes")
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or not all(isinstance(c, str) for c in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("Report classes must be distinct string labels")
    try:
        matrix = np.asarray(metrics.get("confusion_matrix"), dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid report confusion matrix") from error
    if (
        matrix.shape != (len(classes), len(classes))
        or not np.isfinite(matrix).all()
        or (matrix < 0).any()
        or not np.equal(matrix, np.floor(matrix)).all()
    ):
        raise ValueError("Confusion counts must be finite nonnegative integers in class order")
    for name in ("accuracy", "balanced_accuracy", "macro_f1"):
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Report {name} must be a finite score in [0,1]")
    if "test_rows" in report and report["test_rows"] != matrix.sum():
        raise ValueError("Report test_rows disagrees with confusion counts")
    return report
