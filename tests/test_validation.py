import pytest
from classification_lab.datasets import load_dataset
from classification_lab.validation import evaluate_cv, stratified_cv


def test_cv_is_reproducible_with_complete_fold_support():
    data = load_dataset("iris")
    assert evaluate_cv(data) == evaluate_cv(data)
    for train, test in stratified_cv(data.y).split(data.X, data.y):
        assert set(train).isdisjoint(test)
        assert len(set(data.y.iloc[test])) == 3


def test_cv_rejects_impossible_fold_count():
    with pytest.raises(ValueError):
        stratified_cv(load_dataset("iris").y, folds=51)
