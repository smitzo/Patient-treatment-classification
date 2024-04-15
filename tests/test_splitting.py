from classification_lab.datasets import load_patient
from classification_lab.splitting import holdout


def test_seeded_split_is_disjoint_and_stratified():
    data = load_patient()
    a, b = holdout(data), holdout(data)
    assert a.X_train.index.equals(b.X_train.index)
    assert set(a.X_train.index).isdisjoint(a.X_test.index)
    assert len(a.X_train) + len(a.X_test) == len(data.frame)
    assert set(a.y_test) == set(a.y_train)
