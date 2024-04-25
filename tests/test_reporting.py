import json
import pytest
from classification_lab.reporting import write_json


def test_bad_report_does_not_destroy_previous_output(tmp_path):
    path = write_json(tmp_path / "nested/report.json", {"value": 1})
    with pytest.raises(ValueError):
        write_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"value": 1}
    assert list(path.parent.iterdir()) == [path]
