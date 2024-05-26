"""Compare algorithms using the same dataset and seeded folds."""

from .validation import evaluate_cv


def compare(data, models=("dummy", "logistic", "forest"), folds=3, seed=42):
    if not models or len(set(models)) != len(models):
        raise ValueError("Choose a nonempty sequence of distinct models")
    results = [evaluate_cv(data, name, folds, seed) for name in models]
    results.sort(key=lambda report: (-report["scores"]["macro_f1"]["mean"], report["model"]))
    return {
        "dataset": data.spec.name,
        "selection_metric": "macro_f1",
        "folds": folds,
        "seed": seed,
        "results": results,
        "scope": "model selection CV; not an unbiased final score",
    }
