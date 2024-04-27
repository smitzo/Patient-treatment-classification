import json
import pytest
from classification_lab.cli import main


def test_train_cli_emits_parseable_report(tmp_path, capsys):
    path = tmp_path / "report.json"
    assert main(["train", "patient", "--model", "dummy", "--output", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["model"] == "dummy"
    assert path.exists()


def test_invalid_dataset_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as error:
        main(["train", "missing"])
    assert error.value.code == 2
    assert "Unknown dataset" in capsys.readouterr().err
