"""Probability calibration without consuming holdout labels."""

from sklearn.calibration import CalibratedClassifierCV
from .experiments import Experiment
from .models import build_pipeline
from .manifests import reproducibility_manifest
from .leakage import split_overlap
from .metrics import classification_metrics
from .splitting import holdout
from .validation import stratified_cv


def train_calibrated(data, model="logistic", method="sigmoid", folds=3, seed=42):
    if method not in ("sigmoid", "isotonic"):
        raise ValueError("Calibration method must be sigmoid or isotonic")
    split = holdout(data, seed=seed)
    estimator = CalibratedClassifierCV(
        build_pipeline(data.spec, model, seed),
        method=method,
        cv=stratified_cv(split.y_train, folds, seed),
        n_jobs=1,
    )
    estimator.fit(split.X_train, split.y_train)
    report = {
        "schema_version": 1,
        "dataset": data.spec.name,
        "model": model,
        "seed": seed,
        "calibration": method,
        "train_rows": len(split.y_train),
        "test_rows": len(split.y_test),
        "metrics": classification_metrics(
            split.y_test, estimator.predict(split.X_test), estimator.classes_
        ),
        "evaluation": "training-fold calibration, reserved holdout",
    }
    report["reproducibility"] = reproducibility_manifest(data, split, seed)
    report["split_overlap"] = split_overlap(split.X_train, split.X_test)
    return Experiment(estimator, report, split)
