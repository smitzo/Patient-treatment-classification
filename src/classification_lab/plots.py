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
