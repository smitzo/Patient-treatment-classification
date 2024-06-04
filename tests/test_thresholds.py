from classification_lab.datasets import load_dataset
from classification_lab.thresholds import select_threshold, train_thresholded


def test_threshold_can_differ_from_half():
    report = select_threshold(["yes", "yes", "no", "no"], [0.4, 0.3, 0.2, 0.1], "yes")
    assert report["threshold"] == 0.3
    assert report["validation_f1"] == 1


def test_threshold_study_accounts_for_three_disjoint_partitions():
    report = train_thresholded(load_dataset("transactions"), "anomaly")
    assert report["fit_rows"] + report["validation_rows"] + report["test_rows"] == 600
