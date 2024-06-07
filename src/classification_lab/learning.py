"""Cross-validated learning curves refit preprocessing at each training size."""

import numpy as np
from sklearn.model_selection import learning_curve
from .models import build_pipeline
from .quality import validate_bundle
from .validation import stratified_cv


def sample_learning_curve(data, sizes=(0.4, 0.7, 1.0), folds=3, seed=42):
    validate_bundle(data)
    if not sizes or any(not 0 < size <= 1 for size in sizes):
        raise ValueError("Training fractions must lie in (0,1]")
    counts, training, validation = learning_curve(
        build_pipeline(data.spec, seed=seed),
        data.X,
        data.y,
        train_sizes=list(sizes),
        cv=stratified_cv(data.y, folds, seed),
        scoring="f1_macro",
        shuffle=True,
        random_state=seed,
        n_jobs=1,
        error_score="raise",
    )
    return {
        "train_rows": counts.tolist(),
        "training_macro_f1": np.mean(training, axis=1).tolist(),
        "validation_macro_f1": np.mean(validation, axis=1).tolist(),
        "validation_std": np.std(validation, axis=1).tolist(),
        "seed": seed,
    }
