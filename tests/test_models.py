from classification_lab.datasets import load_patient
from classification_lab.models import build_pipeline
from classification_lab.splitting import holdout


def test_logistic_pipeline_predicts_patient_labels():
    data = load_patient()
    split = holdout(data)
    model = build_pipeline(data.spec).fit(split.X_train, split.y_train)
    assert set(model.predict(split.X_test)) <= {"in", "out"}
    assert model.predict_proba(split.X_test).shape == (len(split.X_test), 2)
