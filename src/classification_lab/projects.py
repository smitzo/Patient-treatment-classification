"""Domain-specific studies that use the shared experiment contracts."""

from dataclasses import replace
from .data import DatasetBundle
from .datasets import load_dataset
from .experiments import train


def iris_feature_study(seed=42):
    data = load_dataset("iris")
    reports = {}
    for group in ("sepal", "petal"):
        features = tuple(c for c in data.spec.numeric if c.startswith(group))
        spec = replace(data.spec, numeric=features)
        subset = DatasetBundle(data.frame[list(features) + [spec.target]], spec)
        reports[group] = train(subset, seed=seed).report
    return reports


def wine_scaling_study(seed=42):
    from .metrics import classification_metrics
    from .models import build_pipeline
    from .splitting import holdout

    data = load_dataset("wine")
    split = holdout(data, seed=seed)
    reports = {}
    for scaled in (True, False):
        pipeline = build_pipeline(data.spec, seed=seed)
        if not scaled:
            pipeline.set_params(preprocess__numeric__scale="passthrough")
            pipeline.set_params(model__max_iter=20000)
        pipeline.fit(split.X_train, split.y_train)
        reports["scaled" if scaled else "unscaled"] = classification_metrics(
            split.y_test, pipeline.predict(split.X_test), pipeline.classes_
        )
    return reports
