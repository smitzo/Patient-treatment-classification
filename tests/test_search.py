import numpy as np
from classification_lab.datasets import load_dataset
from classification_lab.search import tune


def test_tuned_scaler_uses_only_reserved_training_data():
    result = tune(load_dataset("wine"))
    scaler = (
        result.estimator.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["scale"]
    )
    np.testing.assert_allclose(scaler.mean_, result.split.X_train.mean())
    assert result.report["best_params"]["model__C"] in [0.1, 1.0, 10.0]
