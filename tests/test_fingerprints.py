from classification_lab.datasets import load_dataset
from classification_lab.fingerprints import dataset_fingerprint


def test_fingerprint_detects_content_and_order_changes():
    data = load_dataset("iris")
    original = dataset_fingerprint(data)
    assert original == dataset_fingerprint(load_dataset("iris"))
    data.frame.iloc[0, 0] += 0.1
    assert dataset_fingerprint(data) != original
    data.frame = data.frame.iloc[::-1]
    assert dataset_fingerprint(data) != original
