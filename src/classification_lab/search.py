"""Hyperparameter selection consumes training folds, followed by one holdout score."""

from sklearn.model_selection import GridSearchCV
from .experiments import Experiment
from .manifests import reproducibility_manifest
from .metrics import classification_metrics
from .models import build_pipeline
from .splitting import holdout
from .validation import stratified_cv


def tune(data, model="logistic", folds=3, seed=42, test_size=0.25):
    grids = {
        "logistic": {"model__C": [0.1, 1.0, 10.0]},
        "forest": {"model__max_depth": [6, None], "model__min_samples_leaf": [1, 3]},
    }
    if model not in grids:
        raise ValueError("Tuning supports logistic and forest models")
    split = holdout(data, test_size, seed)
    search = GridSearchCV(
        build_pipeline(data.spec, model, seed),
        grids[model],
        cv=stratified_cv(split.y_train, folds, seed),
        scoring="f1_macro",
        n_jobs=1,
        error_score="raise",
        refit=True,
    )
    search.fit(split.X_train, split.y_train)
    estimator = search.best_estimator_
    report = {
        "schema_version": 1,
        "dataset": data.spec.name,
        "model": model,
        "seed": seed,
        "test_size": test_size,
        "train_rows": len(split.y_train),
        "test_rows": len(split.y_test),
        "best_params": search.best_params_,
        "selection_cv_macro_f1": float(search.best_score_),
        "metrics": classification_metrics(
            split.y_test, estimator.predict(split.X_test), estimator.classes_
        ),
        "evaluation": "training-only grid search, single reserved holdout",
    }
    report["reproducibility"] = reproducibility_manifest(data, split, seed)
    return Experiment(estimator, report, split)
