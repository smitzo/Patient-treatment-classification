import pandas as pd
from classification_lab.leakage import split_overlap


def test_duplicate_features_detected_despite_different_indices():
    train = pd.DataFrame({"x": [1, 2, 2], "category": ["a", "b", "b"]}, index=[1, 2, 3])
    test = pd.DataFrame({"x": [2, 3], "category": ["b", "c"]}, index=[20, 30])
    report = split_overlap(train, test)
    assert report["shared_feature_patterns"] == 1
    assert report["affected_train_rows"] == 2
    assert report["affected_test_rows"] == 1
