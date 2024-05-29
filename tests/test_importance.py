from classification_lab.datasets import load_dataset
from classification_lab.experiments import train
from classification_lab.importance import heldout_importance


def test_importance_preserves_raw_feature_names():
    result = train(load_dataset("churn"))
    a = heldout_importance(result, repeats=2)
    assert a == heldout_importance(result, repeats=2)
    assert {r["feature"] for r in a} == set(result.split.X_test.columns)
    assert "label" not in {r["feature"] for r in a}
