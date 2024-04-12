import numpy as np
import pandas as pd
import pytest
from classification_lab.data import DatasetBundle, DatasetSpec
from classification_lab.quality import validate_bundle


@pytest.mark.parametrize("values", [[1, np.inf], [None, None], ["bad", "2"]])
def test_invalid_numeric_features_rejected(values):
    data = DatasetBundle(pd.DataFrame({"x": values, "y": [0, 1]}), DatasetSpec("toy", "y", ("x",)))
    with pytest.raises(ValueError):
        validate_bundle(data)


def test_missing_values_counted_without_imputation():
    data = DatasetBundle(
        pd.DataFrame({"x": [1, np.nan], "y": [0, 1]}), DatasetSpec("toy", "y", ("x",))
    )
    assert validate_bundle(data)["missing"] == {"x": 1}
    assert data.X.x.isna().sum() == 1
