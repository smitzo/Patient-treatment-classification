"""Domain-specific studies that use the shared experiment contracts."""

from dataclasses import replace
from .data import DatasetBundle
from .datasets import load_dataset
from .experiments import train


def iris_feature_study(seed=42):
    data = load_dataset("iris")
    reports = {}
    for group in ("sepal", "petal"):
        features = tuple(c for c in data.spec.numeric if c.startswith(group))
        spec = replace(data.spec, numeric=features)
        subset = DatasetBundle(data.frame[list(features) + [spec.target]], spec)
        reports[group] = train(subset, seed=seed).report
    return reports


def wine_scaling_study(seed=42):
    from .metrics import classification_metrics
    from .models import build_pipeline
    from .splitting import holdout

    data = load_dataset("wine")
    split = holdout(data, seed=seed)
    reports = {}
    for scaled in (True, False):
        pipeline = build_pipeline(data.spec, seed=seed)
        if not scaled:
            pipeline.set_params(preprocess__numeric__scale="passthrough")
            pipeline.set_params(model__max_iter=20000)
        pipeline.fit(split.X_train, split.y_train)
        reports["scaled" if scaled else "unscaled"] = classification_metrics(
            split.y_test, pipeline.predict(split.X_test), pipeline.classes_
        )
    return reports


def cancer_detection_study(seed=42):
    """Descriptive benchmark only; no clinical validity is established."""
    from sklearn.metrics import confusion_matrix

    result = train(load_dataset("breast_cancer"), seed=seed)
    truth = result.split.y_test == "malignant"
    predicted = result.estimator.predict(result.split.X_test) == "malignant"
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[False, True]).ravel()
    return {
        "positive_class": "malignant",
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "false_negatives": int(fn),
        "report": result.report,
        "scope": "educational benchmark, not clinical validation",
    }


def digit_error_study(seed=42):
    from collections import Counter

    result = train(load_dataset("digits"), seed=seed)
    pairs = Counter(
        (str(a), str(b))
        for a, b in zip(result.split.y_test, result.estimator.predict(result.split.X_test))
        if a != b
    )
    return {
        "report": result.report,
        "confusions": [
            {"actual": a, "predicted": b, "count": count} for (a, b), count in pairs.most_common()
        ],
    }


def category_error_study(dataset, column, seed=42):
    data = load_dataset(dataset)
    if column not in data.spec.categorical:
        raise ValueError("Breakdown must use a categorical predictor")
    result = train(data, seed=seed)
    predicted = result.estimator.predict(result.split.X_test)
    correct = result.split.y_test.to_numpy() == predicted
    groups = result.split.X_test[column]
    return {
        "report": result.report,
        "breakdown": {
            str(group): {
                "support": int((groups == group).sum()),
                "accuracy": float(correct[(groups == group).to_numpy()].mean()),
            }
            for group in sorted(groups.unique())
        },
    }


def churn_contract_study(seed=42):
    return category_error_study("churn", "contract", seed)
