from classification_lab.datasets import load_dataset


def test_iris_is_offline_balanced_multiclass():
    data = load_dataset("iris")
    assert data.frame.shape == (150, 5)
    assert data.y.value_counts().tolist() == [50, 50, 50]


def test_wine_contains_thirteen_numeric_features():
    data = load_dataset("wine")
    assert data.X.shape == (178, 13)
    assert data.y.nunique() == 3
