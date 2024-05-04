from classification_lab.datasets import load_dataset


def test_iris_is_offline_balanced_multiclass():
    data = load_dataset("iris")
    assert data.frame.shape == (150, 5)
    assert data.y.value_counts().tolist() == [50, 50, 50]


def test_wine_contains_thirteen_numeric_features():
    data = load_dataset("wine")
    assert data.X.shape == (178, 13)
    assert data.y.nunique() == 3


def test_breast_cancer_preserves_named_outcomes():
    data = load_dataset("breast_cancer")
    assert data.X.shape == (569, 30)
    assert set(data.y) == {"malignant", "benign"}
