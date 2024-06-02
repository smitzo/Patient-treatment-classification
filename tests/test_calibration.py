import numpy as np
from classification_lab.calibration import train_calibrated
from classification_lab.datasets import load_dataset


def test_calibration_probabilities_sum_to_one_on_unseen_rows():
    result = train_calibrated(load_dataset("breast_cancer"))
    probability = result.estimator.predict_proba(result.split.X_test)
    np.testing.assert_allclose(probability.sum(axis=1), 1)
    assert result.report["train_rows"] + result.report["test_rows"] == 569
