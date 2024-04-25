import pytest
from classification_lab.metrics import classification_metrics


def test_metrics_keep_requested_class_order_and_minority_failure():
    metrics = classification_metrics(["a", "a", "a", "b"], ["a"] * 4, ["b", "a"])
    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["confusion_matrix"] == [[0, 1], [0, 3]]
    assert metrics["per_class"]["b"]["recall"] == 0


def test_unknown_label_is_not_silently_ignored():
    with pytest.raises(ValueError):
        classification_metrics(["a"], ["b"], ["a"])
