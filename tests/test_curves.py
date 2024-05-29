import json
import pytest
from classification_lab.curves import binary_curves


def test_positive_probability_column_follows_declared_order():
    report = binary_curves(
        ["yes", "no", "yes", "no"],
        [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]],
        ["yes", "no"],
        "yes",
    )
    assert report["roc_auc"] == 1
    json.dumps(report, allow_nan=False)


def test_multiclass_curves_rejected():
    with pytest.raises(ValueError):
        binary_curves(["a", "b"], [[0.5, 0.5]] * 2, ["a", "b", "c"], "a")
