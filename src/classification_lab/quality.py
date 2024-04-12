"""Schema checks and descriptive quality diagnostics."""

import numpy as np
import pandas as pd
from .data import DatasetBundle


def validate_bundle(data: DatasetBundle) -> dict:
    spec, frame = data.spec, data.frame
    if set(frame.columns) != set(spec.features + (spec.target,)) or frame.empty:
        raise ValueError("Dataset does not match its schema")
    if not frame.index.is_unique:
        raise ValueError("Dataset index must be unique")
    if frame[spec.target].isna().any() or frame[spec.target].nunique() < 2:
        raise ValueError("Targets must be present with at least two classes")
    for column in spec.numeric:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series) or np.isinf(series.to_numpy(float)).any():
            raise ValueError(f"{column}: expected finite numeric values or missing values")
        if series.isna().all():
            raise ValueError(f"{column}: entirely missing feature")
    for column in spec.categorical:
        present = frame[column].dropna()
        if present.empty or not present.map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column}: expected categorical strings")
    if spec.text and (
        frame[spec.text].isna().any()
        or not frame[spec.text].map(lambda v: isinstance(v, str) and bool(v.strip())).all()
    ):
        raise ValueError("Text must contain nonempty strings")
    return {
        "rows": len(frame),
        "features": len(spec.features),
        "missing": {c: int(frame[c].isna().sum()) for c in spec.features},
        "duplicate_rows": int(frame.duplicated().sum()),
        "classes": {str(k): int(v) for k, v in frame[spec.target].value_counts().items()},
    }
