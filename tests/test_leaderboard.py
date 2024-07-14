import pytest
from classification_lab.datasets import load_dataset
from classification_lab.experiments import train
from classification_lab.leaderboard import leaderboard
from classification_lab.reporting import write_json


def test_leaderboard_rejects_different_holdout_population(tmp_path):
    paths = [
        write_json(tmp_path / f"{seed}.json", train(load_dataset("iris"), seed=seed).report)
        for seed in [1, 2]
    ]
    with pytest.raises(ValueError, match="heldout population"):
        leaderboard(paths)


def test_shared_holdout_models_can_be_compared(tmp_path):
    paths = [
        write_json(tmp_path / f"{model}.json", train(load_dataset("iris"), model=model).report)
        for model in ["dummy", "logistic"]
    ]
    assert leaderboard(paths)["rows"][0]["model"] == "logistic"
