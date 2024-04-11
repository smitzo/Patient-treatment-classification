import pytest
from classification_lab.csv_io import load_csv
from classification_lab.data import DatasetSpec


@pytest.mark.parametrize("content", ["x,x,y\n1,2,a\n2,3,b\n", "x,y\n1,a\n2,\n", "x,y\n1,a\n2,a\n"])
def test_ambiguous_csv_rejected(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_csv(path, DatasetSpec("toy", "y", ("x",)))
