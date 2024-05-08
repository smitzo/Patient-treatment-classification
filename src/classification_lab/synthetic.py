"""Toy generators; no records describe real people, devices or traffic."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


GENERATORS = {}


def validate_size(rows):
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 40:
        raise ValueError("Synthetic datasets require an integer row count of at least 40")


def draw_labels(rng, score, positive, negative):
    probability = 1 / (1 + np.exp(-np.clip(score, -30, 30)))
    return np.where(rng.random(len(score)) < probability, positive, negative)


def generic_tabular(rows=480, features=6, classes=2, seed=42):
    validate_size(rows)
    if features < 3 or classes < 2:
        raise ValueError("Need at least three features and two classes")
    informative = max(2, int(np.ceil(np.log2(classes))))
    if informative > features:
        raise ValueError("Too many classes for the requested feature count")
    X, y = make_classification(
        n_samples=rows,
        n_features=features,
        n_informative=informative,
        n_redundant=0,
        n_classes=classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    frame = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(features)])
    frame["label"] = y.astype(str)
    return frame


def generate(name, rows=600, seed=42):
    validate_size(rows)
    if name not in GENERATORS:
        raise ValueError(f"Unknown generator {name!r}")
    return GENERATORS[name](rows, seed)
