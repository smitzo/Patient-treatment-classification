from classification_lab.datasets import load_dataset
from classification_lab.sensitivity import seed_sensitivity


def test_seed_sensitivity_retains_each_holdout_report():
    report = seed_sensitivity(load_dataset("iris"), seeds=(4, 8))
    assert [r["seed"] for r in report["reports"]] == [4, 8]
    assert report["macro_f1_std"] >= 0
