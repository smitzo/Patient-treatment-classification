"""Transformations are fitted by the enclosing estimator, never globally."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import DatasetSpec


def tabular_preprocessor(spec: DatasetSpec) -> ColumnTransformer:
    if spec.text:
        raise ValueError("Use a text pipeline for text datasets")
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(spec.numeric)),
            ("categorical", categorical, list(spec.categorical)),
        ]
    )
