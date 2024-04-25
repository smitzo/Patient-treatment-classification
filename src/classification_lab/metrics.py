"""JSON-native classification metrics with explicit class ordering."""

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_metrics(y_true, y_pred, classes) -> dict:
    classes = list(classes)
    if len(y_true) != len(y_pred) or not len(y_true):
        raise ValueError("Truth and predictions need equal nonzero lengths")
    if not set(y_true).union(y_pred).issubset(set(classes)):
        raise ValueError("Labels outside the declared class order")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        ),
        "classes": [str(c) for c in classes],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=classes,
            target_names=[str(c) for c in classes],
            output_dict=True,
            zero_division=0,
        ),
    }
