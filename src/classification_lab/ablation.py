"""Feature selection diagnostics belong inside training data, not repeated holdout tuning."""

from dataclasses import replace
from .data import DatasetBundle
from .validation import evaluate_cv


def feature_ablation(training_data, drop, folds=3, seed=42):
    spec = training_data.spec
    if spec.text or not drop or not set(drop).issubset(spec.features):
        raise ValueError("Choose known tabular predictors to remove")
    numeric = tuple(c for c in spec.numeric if c not in drop)
    categorical = tuple(c for c in spec.categorical if c not in drop)
    reduced_spec = replace(spec, numeric=numeric, categorical=categorical)
    frame = training_data.frame[list(reduced_spec.features) + [spec.target]].copy()
    reduced = DatasetBundle(frame, reduced_spec)
    return {
        "dropped": list(drop),
        "full": evaluate_cv(training_data, folds=folds, seed=seed),
        "reduced": evaluate_cv(reduced, folds=folds, seed=seed),
        "scope": "training-only feature-selection CV; reserve external holdout separately",
    }
