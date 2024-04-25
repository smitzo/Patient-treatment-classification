from classification_lab.datasets import load_patient
from classification_lab.experiments import train


def test_training_repeats_for_identical_seed():
    a = train(load_patient(), "logistic", seed=19)
    b = train(load_patient(), "logistic", seed=19)
    assert a.report == b.report
    assert a.report["train_rows"] + a.report["test_rows"] == 4412
