"""Shared experiment path for scripts and command-line tools."""

from dataclasses import dataclass
from .leakage import split_overlap
from .metrics import classification_metrics
from .models import build_pipeline
from .splitting import Split, holdout


@dataclass
class Experiment:
    estimator: object
    report: dict
    split: Split


def train(data, model="logistic", seed=42, test_size=0.25):
    split = holdout(data, test_size, seed)
    estimator = build_pipeline(data.spec, model, seed)
    estimator.fit(split.X_train, split.y_train)
    metrics = classification_metrics(
        split.y_test, estimator.predict(split.X_test), estimator.classes_
    )
    report = {
        "schema_version": 1,
        "dataset": data.spec.name,
        "model": model,
        "seed": seed,
        "test_size": test_size,
        "train_rows": len(split.y_train),
        "test_rows": len(split.y_test),
        "metrics": metrics,
        "evaluation": "single stratified holdout; educational, not external validation",
    }
    report["split_overlap"] = split_overlap(split.X_train, split.X_test)
    return Experiment(estimator, report, split)
