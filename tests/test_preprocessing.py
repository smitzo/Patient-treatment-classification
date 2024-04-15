import pandas as pd
from classification_lab.data import DatasetSpec
from classification_lab.preprocessing import tabular_preprocessor


def test_scaler_does_not_learn_heldout_outlier_and_accepts_unseen_category():
    spec = DatasetSpec("toy", "y", ("x",), ("category",))
    pre = tabular_preprocessor(spec)
    pre.fit(pd.DataFrame({"x": [1.0, 3.0], "category": ["a", "b"]}))
    transformed = pre.transform(pd.DataFrame({"x": [1000.0], "category": ["new"]}))
    assert pre.named_transformers_["numeric"].named_steps["scale"].mean_[0] == 2
    assert transformed.shape == (1, 3)
