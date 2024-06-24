"""Prediction input excludes the target and rejects ambiguous columns."""

import numpy as np
import pandas as pd
from .probabilities import validate_probabilities


def validate_features(frame, spec):
    if frame.empty or frame.columns.duplicated().any() or set(frame.columns) != set(spec.features):
        raise ValueError(
            f"Prediction columns must be exactly {list(spec.features)} with nonzero rows"
        )
    frame = frame.loc[:, list(spec.features)].copy()
    for column in spec.numeric:
        if (
            not pd.api.types.is_numeric_dtype(frame[column])
            or np.isinf(frame[column].to_numpy(float)).any()
        ):
            raise ValueError(f"{column}: expected finite numeric values or missing values")
    for column in spec.categorical:
        if not frame[column].dropna().map(lambda v: isinstance(v, str)).all():
            raise ValueError(f"{column}: categorical values must be strings")
    if (
        spec.text
        and not frame[spec.text].map(lambda v: isinstance(v, str) and bool(v.strip())).all()
    ):
        raise ValueError("Prediction text must contain nonempty strings")
    return frame


def predict_frame(artifact, frame):
    X = validate_features(frame, artifact.spec)
    classes = [str(c) for c in artifact.estimator.classes_]
    probability = validate_probabilities(artifact.estimator.predict_proba(X), classes, rows=len(X))
    result = pd.DataFrame({"prediction": artifact.estimator.predict(X)}, index=X.index)
    for i, label in enumerate(classes):
        result[f"probability:{label}"] = probability[:, i]
    return result
