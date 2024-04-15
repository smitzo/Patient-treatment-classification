from classification_lab.datasets import load_patient


def test_original_patient_contract():
    data = load_patient()
    assert data.frame.shape == (4412, 11)
    assert len(data.spec.numeric) == 9
    assert set(data.y) == {"in", "out"}
    assert "SOURCE" not in data.X.columns
