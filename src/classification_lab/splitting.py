"""Explicit split boundaries with preserved row indices."""

from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from .data import DatasetBundle
from .quality import validate_bundle


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def holdout(data: DatasetBundle, test_size: float = 0.25, seed: int = 42) -> Split:
    validate_bundle(data)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    if data.y.value_counts().min() < 2:
        raise ValueError("Each class needs at least two rows for stratification")
    values = train_test_split(
        data.X, data.y, test_size=test_size, random_state=seed, stratify=data.y
    )
    split = Split(*values)
    if set(split.y_train) != set(data.y) or set(split.y_test) != set(data.y):
        raise ValueError("Both partitions must contain every class")
    return split
