from classification_lab.datasets import load_dataset
from classification_lab.experiments import train


def test_manifests_reproduce_and_identify_changed_partitions():
    data = load_dataset("iris")
    a = train(data, seed=1).report["reproducibility"]
    b = train(data, seed=2).report["reproducibility"]
    assert a["dataset_fingerprint"] == b["dataset_fingerprint"]
    assert a["train_index_fingerprint"] != b["train_index_fingerprint"]
    assert a["packages"]["scikit-learn"]
