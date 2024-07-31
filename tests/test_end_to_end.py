"""Cross-component regressions at schema, persistence and command boundaries."""

from dataclasses import replace
import json
from pathlib import Path
import pickle
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from classification_lab.artifacts import load_artifact, save_artifact
from classification_lab.cli import main
from classification_lab.csv_io import load_csv, read_checked_csv
from classification_lab.data import DatasetSpec
from classification_lab.datasets import LOADERS, load_dataset
from classification_lab.experiments import train
from classification_lab.metrics import classification_metrics
from classification_lab.models import build_pipeline
from classification_lab.quality import validate_bundle
from classification_lab.report_contracts import validate_report


@pytest.mark.parametrize("name", sorted(LOADERS))
def test_every_registered_project_has_valid_metrics_and_manifest(name):
    result = train(load_dataset(name))
    validate_report(result.report)
    probability = result.estimator.predict_proba(result.split.X_test)
    np.testing.assert_allclose(probability.sum(axis=1), 1)
    assert result.report["reproducibility"]["test_index_fingerprint"]


@pytest.mark.parametrize(
    "content",
    [
        "x,y\n1,a,extra\n2,b,extra\n",
        "x,y\n1\n2,b\n",
        "x,x\n1,2\n",
        "x,\n1,2\n",
        "",
        'x,y\n1,"unterminated',
    ],
)
def test_bad_csv_does_not_infer_an_index_or_truncate_rows(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_checked_csv(path)


def test_csv_preserves_leading_zero_target_and_category(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("x,category,label\n1,001,01\n2,002,02\n")
    data = load_csv(path, DatasetSpec("toy", "label", ("x",), ("category",)))
    assert data.y.tolist() == ["01", "02"]
    assert data.X.category.tolist() == ["001", "002"]


def test_nullable_features_share_imputation_behavior_with_none():
    spec = DatasetSpec("toy", "label", ("x",), ("category",))
    frame = pd.DataFrame(
        {
            "x": pd.Series([1, None, 3, 4], dtype="Float64"),
            "category": pd.Series(["a", None, "b", "a"], dtype="string"),
        }
    )
    pipeline = build_pipeline(spec).fit(frame, ["no", "yes", "no", "yes"])
    prediction = frame.copy()
    prediction["category"] = [None, "new", "a", "b"]
    assert len(pipeline.predict(prediction)) == 4


def test_metric_labels_cannot_overwrite_per_class_values():
    result = classification_metrics(
        ["accuracy", "macro avg"], ["accuracy", "accuracy"], ["accuracy", "macro avg"]
    )
    assert result["per_class"]["accuracy"]["support"] == 1
    assert result["per_class"]["macro avg"]["recall"] == 0
    with pytest.raises(ValueError):
        classification_metrics(["a"], ["a"], ["a", "a"])


def test_duplicate_dataframe_columns_fail_before_feature_access():
    data = load_dataset("iris")
    data.frame = pd.concat([data.frame, data.frame.iloc[:, [0]]], axis=1)
    with pytest.raises(ValueError, match="schema"):
        validate_bundle(data)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"not a pickle",
        pickle.dumps({"metadata": None, "estimator": None}),
        pickle.dumps({"metadata": {"format_version": 1}, "estimator": None}),
    ],
)
def test_corrupt_trusted_models_fail_with_actionable_error(tmp_path, payload):
    path = tmp_path / "bad.pkl"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        load_artifact(path, trusted=True)


def test_artifact_rejects_wrong_feature_schema_before_output(tmp_path):
    data = load_dataset("iris")
    result = train(data)
    path = tmp_path / "bad.pkl"
    with pytest.raises(ValueError, match="feature schema"):
        save_artifact(result, replace(data.spec, numeric=tuple(reversed(data.spec.numeric))), path)
    assert not path.exists()


@pytest.mark.parametrize("name", ["spam", "churn"])
def test_text_and_mixed_tabular_artifacts_roundtrip(tmp_path, name):
    data = load_dataset(name)
    result = train(data)
    path = save_artifact(result, data.spec, tmp_path / f"{name}.pkl")
    artifact = load_artifact(path, trusted=True)
    np.testing.assert_allclose(
        artifact.estimator.predict_proba(data.X), result.estimator.predict_proba(data.X)
    )


@pytest.mark.parametrize(
    "change", ["missing_metrics", "nan_score", "wrong_shape", "negative_count"]
)
def test_report_validation_rejects_malformed_rank_inputs(change):
    report = train(load_dataset("iris")).report
    if change == "missing_metrics":
        report.pop("metrics")
    elif change == "nan_score":
        report["metrics"]["macro_f1"] = float("nan")
    elif change == "wrong_shape":
        report["metrics"]["confusion_matrix"] = [[1]]
    else:
        report["metrics"]["confusion_matrix"][0][0] = -1
    with pytest.raises(ValueError):
        validate_report(report)


def test_cli_corrupt_model_has_exit_two_without_traceback(tmp_path, capsys):
    path = tmp_path / "bad.pkl"
    path.write_bytes(b"invalid pickle")
    with pytest.raises(SystemExit) as error:
        main(
            [
                "predict",
                str(path),
                "unused.csv",
                "--trust-artifact",
                "--output-csv",
                str(tmp_path / "output.csv"),
            ]
        )
    assert error.value.code == 2
    assert "Unreadable trusted model" in capsys.readouterr().err


def test_module_entrypoint_executes_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "classification_lab", "train", "iris", "--model", "dummy"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout)["dataset"] == "iris"
    assert "Traceback" not in result.stderr


def test_all_standalone_examples_execute_as_scripts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    names = [p for p in sorted((root / "examples").glob("*.py")) if p.name != "run_all.py"]
    assert len(names) == 11
    for script in names:
        command = [sys.executable, str(script)]
        if script.name == "patient.py":
            command += ["--model", "dummy", "--data-path", str(root / "patient_data.csv")]
        result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, check=True)
        assert isinstance(json.loads(result.stdout), dict), script.name


def test_generated_csv_can_be_trained_with_its_registered_schema(tmp_path, capsys):
    path = tmp_path / "new-churn.csv"
    main(["generate", "churn", "--rows", "80", "--seed", "11", "--output-csv", str(path)])
    capsys.readouterr()
    data = load_csv(path, load_dataset("churn").spec)
    assert train(data).report["train_rows"] == 60
