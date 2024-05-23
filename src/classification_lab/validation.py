"""Training-fold evaluation helpers; every fold fits a fresh pipeline."""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from .models import build_pipeline
from .quality import validate_bundle


def stratified_cv(y, folds=3, seed=42):
    if folds < 2 or y.value_counts().min() < folds:
        raise ValueError("Each class needs at least one row per fold, with folds >= 2")
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def evaluate_cv(data, model="logistic", folds=3, seed=42):
    validate_bundle(data)
    result = cross_validate(
        build_pipeline(data.spec, model, seed),
        data.X,
        data.y,
        cv=stratified_cv(data.y, folds, seed),
        scoring={"balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro"},
        n_jobs=1,
        error_score="raise",
    )
    return {
        "dataset": data.spec.name,
        "model": model,
        "folds": folds,
        "seed": seed,
        "scope": "cross-validation; no external validation",
        "scores": {
            key[5:]: {
                "folds": values.tolist(),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            for key, values in result.items()
            if key.startswith("test_")
        },
    }
