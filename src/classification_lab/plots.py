"""Standalone scientific plots using noninteractive figures."""

from pathlib import Path
import numpy as np
from matplotlib.figure import Figure


def save_figure(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, bbox_inches="tight", dpi=140)
    finally:
        figure.clear()
    return path


def confusion_plot(metrics, path, normalize=False):
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    classes = metrics["classes"]
    if (
        matrix.shape != (len(classes), len(classes))
        or (matrix < 0).any()
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Confusion matrix must be finite, nonnegative and square in class order")
    if normalize:
        totals = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals != 0)
    figure = Figure(figsize=(6, 5))
    axis = figure.subplots()
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted",
        ylabel="Actual",
        title="Confusion matrix",
    )
    for row in range(len(classes)):
        for column in range(len(classes)):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}" if normalize else f"{matrix[row, column]:g}",
                ha="center",
                va="center",
            )
    return save_figure(figure, path)


def probability_curve_plot(curves, path):
    figure = Figure(figsize=(10, 4))
    roc_axis, pr_axis = figure.subplots(1, 2)
    roc = curves["roc"]
    roc_axis.plot(roc["fpr"], roc["tpr"], label=f"AUC {curves['roc_auc']:.3f}")
    roc_axis.plot([0, 1], [0, 1], "--", color="gray")
    roc_axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"ROC: {curves['positive_class']}",
    )
    roc_axis.legend()
    pr = curves["precision_recall"]
    pr_axis.plot(pr["recall"], pr["precision"], label=f"AP {curves['average_precision']:.3f}")
    pr_axis.set(
        xlabel="Recall", ylabel="Precision", title="Precision-recall", xlim=(0, 1), ylim=(0, 1.05)
    )
    pr_axis.legend()
    return save_figure(figure, path)


def importance_plot(rows, path, limit=20):
    if not rows or limit < 1:
        raise ValueError("Need importance rows and a positive feature limit")
    rows = sorted(rows, key=lambda row: row["mean"], reverse=True)[:limit][::-1]
    figure = Figure(figsize=(8, max(3, len(rows) * 0.35)))
    axis = figure.subplots()
    axis.barh(
        [r["feature"] for r in rows], [r["mean"] for r in rows], xerr=[r["std"] for r in rows]
    )
    axis.axvline(0, color="gray", linewidth=1)
    axis.set(xlabel="Decrease in heldout balanced accuracy", title="Permutation importance")
    return save_figure(figure, path)
