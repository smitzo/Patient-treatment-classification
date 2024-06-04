"""Threshold selection uses a separate validation partition inside training data."""

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from .metrics import classification_metrics
from .models import build_pipeline
from .splitting import holdout


def select_threshold(y_validation, positive_probability, positive):
    score = np.asarray(positive_probability, dtype=float)
    truth = np.asarray(y_validation) == positive
    if score.shape != truth.shape or not len(score) or not np.isfinite(score).all():
        raise ValueError("Validation probabilities must be finite and aligned")
    if np.any((score < 0) | (score > 1)) or len(np.unique(truth)) != 2:
        raise ValueError("Need probabilities in [0, 1] and both classes")
    candidates = np.unique(np.r_[0.0, score, 1.0])
    ranked = [
        (
            float(f1_score(truth, score >= threshold, zero_division=0)),
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        for threshold in candidates
    ]
    best, _, threshold = max(ranked)
    return {"threshold": threshold, "validation_f1": best, "validation_rows": len(score)}


def train_thresholded(data, positive, seed=42):
    split = holdout(data, seed=seed)
    if data.y.nunique() != 2 or positive not in set(data.y):
        raise ValueError("Threshold study requires binary data and a positive label")
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        split.X_train, split.y_train, test_size=0.25, stratify=split.y_train, random_state=seed
    )
    estimator = build_pipeline(data.spec, seed=seed).fit(X_fit, y_fit)
    classes = list(estimator.classes_)
    index = classes.index(positive)
    selection = select_threshold(
        y_validation, estimator.predict_proba(X_validation)[:, index], positive
    )
    probability = estimator.predict_proba(split.X_test)[:, index]
    negative = next(c for c in classes if c != positive)
    prediction = np.where(probability >= selection["threshold"], positive, negative)
    return {
        "dataset": data.spec.name,
        "positive_class": str(positive),
        **selection,
        "fit_rows": len(y_fit),
        "test_rows": len(split.y_test),
        "metrics": classification_metrics(split.y_test, prediction, classes),
        "scope": "validation-selected threshold, untouched outer holdout; no post-selection refit",
    }
