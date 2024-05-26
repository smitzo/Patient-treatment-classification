from classification_lab.comparison import compare
from classification_lab.datasets import load_dataset


def test_comparison_uses_same_seed_and_orders_macro_f1():
    report = compare(load_dataset("iris"), models=("dummy", "logistic"), seed=8)
    assert report["results"][0]["model"] == "logistic"
    assert {r["seed"] for r in report["results"]} == {8}
    assert {r["folds"] for r in report["results"]} == {3}
