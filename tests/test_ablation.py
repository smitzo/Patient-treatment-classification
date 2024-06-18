import pytest
from classification_lab.ablation import feature_ablation
from classification_lab.datasets import load_dataset


def test_ablation_does_not_mutate_original_dataset():
    data = load_dataset("iris")
    original = data.X.columns.tolist()
    report = feature_ablation(data, drop=[original[0]])
    assert report["dropped"] == [original[0]]
    assert data.X.columns.tolist() == original
    with pytest.raises(ValueError):
        feature_ablation(data, drop=original)
