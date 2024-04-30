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
