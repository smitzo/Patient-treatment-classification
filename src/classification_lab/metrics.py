"""Class-aware metrics without reserved-name collisions or undefined values."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(y_true, y_pred, classes):
    classes = list(classes)
    truth, prediction = np.asarray(y_true), np.asarray(y_pred)
    if (
        len(classes) < 2
        or len(set(classes)) != len(classes)
        or len(set(map(str, classes))) != len(classes)
    ):
        raise ValueError("Declare at least two distinct unambiguous classes")
    if truth.ndim != 1 or prediction.ndim != 1 or truth.shape != prediction.shape or not len(truth):
        raise ValueError("Truth and predictions need aligned nonempty one-dimensional labels")
    if pd.isna(truth).any() or pd.isna(prediction).any():
        raise ValueError("Evaluation labels cannot be missing")
    if not set(truth).union(prediction).issubset(set(classes)):
        raise ValueError("Labels outside the declared class order")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=classes, zero_division=0
    )
    per_class = {
        str(label): {
            "precision": float(p),
            "recall": float(r),
            "f1-score": float(f),
            "support": int(n),
        }
        for label, p, r, f, n in zip(classes, precision, recall, f1, support)
    }
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(np.mean(recall[support > 0])),
        "macro_f1": float(
            f1_score(truth, prediction, labels=classes, average="macro", zero_division=0)
        ),
        "classes": [str(c) for c in classes],
        "confusion_matrix": confusion_matrix(truth, prediction, labels=classes).tolist(),
        "per_class": per_class,
    }
