"""Bounded deterministic factories for text and tabular classification."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .preprocessing import tabular_preprocessor


MODEL_NAMES = ["logistic", "forest", "dummy"]


def build_pipeline(spec, model: str = "logistic", seed: int = 42):
    factories = {
        "logistic": lambda: LogisticRegression(max_iter=2000, random_state=seed),
        "forest": lambda: RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
        "dummy": lambda: DummyClassifier(strategy="most_frequent", random_state=seed),
    }
    if model not in factories:
        raise ValueError(f"Unknown model {model!r}; choose from {MODEL_NAMES}")
    if spec.text:
        preprocessor = ColumnTransformer(
            [("text", TfidfVectorizer(ngram_range=(1, 2), max_features=10000), spec.text)]
        )
    else:
        preprocessor = tabular_preprocessor(spec)
    return Pipeline([("preprocess", preprocessor), ("model", factories[model]())])
