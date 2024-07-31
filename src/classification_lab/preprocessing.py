"""Transformations are fitted by the enclosing estimator, never globally."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from .data import DatasetSpec


def normalize_missing(frame):
    """Normalize None and pandas nullable scalars before sklearn imputation."""
    return frame.astype(object).where(pd.notna(frame), np.nan)


def tabular_preprocessor(spec: DatasetSpec) -> ColumnTransformer:
    if spec.text:
        raise ValueError("Use a text pipeline for text datasets")
    numeric = Pipeline(
        [
            ("normalize", FunctionTransformer(normalize_missing, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("normalize", FunctionTransformer(normalize_missing, feature_names_out="one-to-one")),
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
