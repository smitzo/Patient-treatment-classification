"""Percentile intervals conditional on fixed predictions, not retraining uncertainty."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def bootstrap_interval(y_true, y_pred, metric="accuracy", samples=500, confidence=0.95, seed=42):
    truth, prediction = np.asarray(y_true), np.asarray(y_pred)
    if len(truth) != len(prediction) or len(truth) < 2 or samples < 20 or not 0 < confidence < 1:
        raise ValueError("Need aligned observations, >=20 samples and confidence in (0,1)")
    classes = np.unique(np.r_[truth, prediction])
    if metric == "accuracy":
        score = accuracy_score
    elif metric == "macro_f1":

        def score(a, b):
            return f1_score(a, b, labels=classes, average="macro", zero_division=0)
    else:
        raise ValueError("Choose accuracy or macro_f1")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        index = rng.integers(0, len(truth), len(truth))
        values.append(score(truth[index], prediction[index]))
    low, high = np.quantile(values, [(1 - confidence) / 2, (1 + confidence) / 2])
    return {
        "metric": metric,
        "estimate": float(score(truth, prediction)),
        "low": float(low),
        "high": float(high),
        "confidence": confidence,
        "samples": samples,
        "seed": seed,
        "scope": "paired observation bootstrap conditional on fixed predictions",
    }
