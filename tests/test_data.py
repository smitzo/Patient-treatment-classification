import pandas as pd
import pytest
from classification_lab.data import DatasetBundle, DatasetSpec


def test_target_cannot_be_feature():
    with pytest.raises(ValueError, match="distinct"):
        DatasetSpec("bad", "label", numeric=("label",))


def test_feature_access_isolated_from_frame():
    data = DatasetBundle(pd.DataFrame({"x": [1], "y": [0]}), DatasetSpec("toy", "y", ("x",)))
    features = data.X
    features.iloc[0, 0] = 99
    assert data.frame.x.iloc[0] == 1
    assert "y" not in features
