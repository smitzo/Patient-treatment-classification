from classification_lab.examples import run_example


def test_patient_example_uses_shared_experiment(capsys):
    result = run_example("patient", ["--model", "dummy"])
    assert result.report["dataset"] == "patient"
    assert result.report["test_rows"] > 0
    assert "single stratified holdout" in capsys.readouterr().out
