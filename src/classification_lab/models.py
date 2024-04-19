"""Bounded deterministic model factories."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .preprocessing import tabular_preprocessor

MODEL_NAMES = ["logistic"]


def build_pipeline(spec, model: str = "logistic", seed: int = 42):
    if model != "logistic":
        raise ValueError(f"Unknown model {model!r}")
    estimator = LogisticRegression(max_iter=2000, random_state=seed)
    return Pipeline([("preprocess", tabular_preprocessor(spec)), ("model", estimator)])
