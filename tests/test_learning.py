from classification_lab.datasets import load_dataset
from classification_lab.learning import sample_learning_curve


def test_learning_curve_increases_training_population():
    report = sample_learning_curve(load_dataset("iris"))
    assert report["train_rows"] == [40, 70, 100]
    assert len(report["validation_macro_f1"]) == 3
