import pytest
from classification_lab.artifacts import ModelArtifact
from classification_lab.datasets import load_dataset
from classification_lab.experiments import train
from classification_lab.prediction import predict_frame


def test_prediction_reorders_columns_and_rejects_target():
    data = load_dataset("iris")
    result = train(data)
    artifact = ModelArtifact(result.estimator, data.spec, {})
    prediction = predict_frame(artifact, data.X.iloc[:3, ::-1])
    assert prediction.index.tolist() == [0, 1, 2]
    assert prediction.columns.tolist()[1:] == [
        f"probability:{c}" for c in result.estimator.classes_
    ]
    with pytest.raises(ValueError):
        predict_frame(artifact, data.frame.iloc[:3])
