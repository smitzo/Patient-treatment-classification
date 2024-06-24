import pytest
from classification_lab.artifacts import load_artifact, save_artifact
from classification_lab.datasets import load_dataset
from classification_lab.experiments import train


def test_trust_is_required_before_any_file_access(tmp_path):
    with pytest.raises(ValueError, match="explicitly trust"):
        load_artifact(tmp_path / "does-not-exist.pkl")


def test_saved_pipeline_roundtrip_preserves_predictions(tmp_path):
    data = load_dataset("iris")
    result = train(data)
    path = save_artifact(result, data.spec, tmp_path / "model.pkl")
    restored = load_artifact(path, trusted=True)
    assert restored.spec == data.spec
    assert (restored.estimator.predict(data.X) == result.estimator.predict(data.X)).all()
    assert path.with_suffix(".pkl.json").exists()
