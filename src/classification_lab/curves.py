"""Binary probability diagnostics with an explicit positive class."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def binary_curves(y_true, probabilities, classes, positive):
    classes = list(classes)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(classes) != 2 or positive not in classes:
        raise ValueError("Binary curves need two classes and a declared positive label")
    if probabilities.shape != (len(y_true), 2) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability matrix")
    truth = np.asarray(y_true) == positive
    if set(np.unique(truth)) != {False, True} or not set(y_true).issubset(set(classes)):
        raise ValueError("Both declared classes must be present")
    score = probabilities[:, classes.index(positive)]
    fpr, tpr, roc_thresholds = roc_curve(truth, score)
    precision, recall, thresholds = precision_recall_curve(truth, score)
    return {
        "positive_class": str(positive),
        "roc_auc": float(roc_auc_score(truth, score)),
        "average_precision": float(average_precision_score(truth, score)),
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": [float(v) if np.isfinite(v) else None for v in roc_thresholds],
        },
        "precision_recall": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        },
    }
