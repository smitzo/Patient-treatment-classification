"""Bounded deterministic model factories."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .preprocessing import tabular_preprocessor

MODEL_NAMES = ["logistic", "forest"]


def build_pipeline(spec, model: str = "logistic", seed: int = 42):
    factories = {
        "logistic": lambda: LogisticRegression(max_iter=2000, random_state=seed),
        "forest": lambda: RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
    }
    if model not in factories:
        raise ValueError(f"Unknown model {model!r}")
    return Pipeline([("preprocess", tabular_preprocessor(spec)), ("model", factories[model]())])
