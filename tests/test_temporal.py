import pandas as pd
from classification_lab.data import DatasetBundle, DatasetSpec
from classification_lab.temporal import temporal_holdout


def test_equal_timestamps_do_not_cross_boundary():
    data = DatasetBundle(
        pd.DataFrame({"x": range(8), "y": ["a", "b"] * 4}), DatasetSpec("toy", "y", ("x",))
    )
    times = ["2020-01-01"] * 2 + ["2020-01-02"] * 3 + ["2020-01-03"] * 3
    split = temporal_holdout(data, times, test_size=0.4)
    assert max(times[i] for i in split.X_train.index) < min(times[i] for i in split.X_test.index)
    assert len(split.X_test) == 6
