import numpy as np
from classification_lab.calibration import train_calibrated
from classification_lab.datasets import load_dataset


def test_calibration_probabilities_sum_to_one_on_unseen_rows():
    result = train_calibrated(load_dataset("breast_cancer"))
    probability = result.estimator.predict_proba(result.split.X_test)
    np.testing.assert_allclose(probability.sum(axis=1), 1)
    assert result.report["train_rows"] + result.report["test_rows"] == 569


def test_calibrated_artifact_has_split_provenance_and_named_features(tmp_path):
    from classification_lab.artifacts import load_artifact, save_artifact

    data = load_dataset("iris")
    result = train_calibrated(data)
    assert result.report["reproducibility"]["dataset_fingerprint"]
    path = save_artifact(result, data.spec, tmp_path / "calibrated.pkl")
    assert load_artifact(path, trusted=True).spec == data.spec
