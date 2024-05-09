import pandas as pd
import pytest
from classification_lab.synthetic import generic_tabular


def test_generic_generation_is_repeatable_and_seed_sensitive():
    a = generic_tabular(seed=3)
    pd.testing.assert_frame_equal(a, generic_tabular(seed=3))
    assert not a.equals(generic_tabular(seed=4))
    assert a.label.nunique() == 2


def test_too_small_generation_rejected():
    with pytest.raises(ValueError):
        generic_tabular(rows=3)


def test_churn_snapshot_matches_generator():
    from classification_lab.datasets import load_dataset
    from classification_lab.synthetic import generate

    pd.testing.assert_frame_equal(load_dataset("churn").frame, generate("churn"))
