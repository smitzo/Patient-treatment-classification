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


# A baseline uses the same preprocessing and split contract as learned models.
def build_dummy_pipeline(spec, seed: int = 42):
    from sklearn.dummy import DummyClassifier

    return Pipeline(
        [
            ("preprocess", tabular_preprocessor(spec)),
            ("model", DummyClassifier(strategy="most_frequent", random_state=seed)),
        ]
    )


MODEL_NAMES.append("dummy")
_original_build_pipeline = build_pipeline


def build_pipeline(spec, model: str = "logistic", seed: int = 42):
    if model == "dummy":
        return build_dummy_pipeline(spec, seed)
    return _original_build_pipeline(spec, model, seed)
