from classification_lab.datasets import load_patient
from classification_lab.models import build_pipeline
from classification_lab.splitting import holdout


def test_logistic_pipeline_predicts_patient_labels():
    data = load_patient()
    split = holdout(data)
    model = build_pipeline(data.spec).fit(split.X_train, split.y_train)
    assert set(model.predict(split.X_test)) <= {"in", "out"}
    assert model.predict_proba(split.X_test).shape == (len(split.X_test), 2)


def test_forest_is_seeded_and_bounded():
    data = load_patient()
    model = build_pipeline(data.spec, "forest", seed=7)
    assert model.named_steps["model"].random_state == 7
    assert model.named_steps["model"].n_jobs == 1
    model.fit(data.X.iloc[:200], data.y.iloc[:200])
    assert len(model.predict(data.X.iloc[:5])) == 5
