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


def test_prediction_cli_requires_trust_then_roundtrips(tmp_path, capsys):
    from classification_lab.datasets import load_dataset

    artifact = tmp_path / "model.pkl"
    main(["train", "iris", "--artifact", str(artifact)])
    capsys.readouterr()
    features = tmp_path / "features.csv"
    load_dataset("iris").X.iloc[:5].to_csv(features, index=False)
    output = tmp_path / "predictions.csv"
    command = ["predict", str(artifact), str(features), "--output-csv", str(output)]
    with pytest.raises(SystemExit):
        main(command)
    main(command + ["--trust-artifact"])
    assert len(output.read_text().splitlines()) == 6


def test_dataset_listing_includes_text_and_tabular(capsys):
    main(["datasets"])
    rows = json.loads(capsys.readouterr().out)["datasets"]
    assert len(rows) == 11
    assert {r["kind"] for r in rows} == {"tabular", "text"}


def test_validation_cli_counts_real_patient_records(capsys):
    main(["validate", "patient"])
    report = json.loads(capsys.readouterr().out)
    assert report["rows"] == 4412
    assert sum(report["classes"].values()) == 4412


def test_comparison_cli_records_selection_metric(capsys):
    main(["compare", "iris", "--models", "dummy", "logistic"])
    report = json.loads(capsys.readouterr().out)
    assert report["selection_metric"] == "macro_f1"
    assert len(report["results"]) == 2


def test_tune_cli_records_training_selection(capsys):
    main(["tune", "iris"])
    report = json.loads(capsys.readouterr().out)
    assert "best_params" in report
    assert "reserved holdout" in report["evaluation"]


def test_generation_cli_matches_requested_size_and_seed(tmp_path, capsys):
    import pandas as pd

    output = tmp_path / "toy.csv"
    main(["generate", "churn", "--rows", "80", "--seed", "9", "--output-csv", str(output)])
    report = json.loads(capsys.readouterr().out)
    assert report["kind"] == "synthetic" and report["seed"] == 9
    assert len(pd.read_csv(output)) == 80


def test_evaluate_cli_uses_explicit_class_order(tmp_path, capsys):
    path = tmp_path / "labels.csv"
    path.write_text("truth,prediction\na,a\nb,a\n")
    main(["evaluate", str(path), "--classes", "b", "a"])
    report = json.loads(capsys.readouterr().out)
    assert report["metrics"]["classes"] == ["b", "a"]
    assert report["metrics"]["accuracy"] == 0.5


def test_inspect_cli_does_not_open_model(tmp_path, capsys):
    path = tmp_path / "dangerous.pkl"
    path.write_bytes(b"not a pickle")
    path.with_suffix(".pkl.json").write_text(
        json.dumps({"format_version": 1, "description": "metadata only"})
    )
    main(["inspect", str(path)])
    assert "model not opened" in json.loads(capsys.readouterr().out)["scope"]
