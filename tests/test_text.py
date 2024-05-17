from classification_lab.datasets import load_dataset
from classification_lab.models import build_pipeline
from classification_lab.experiments import train


def test_text_vocabulary_is_training_only():
    data = load_dataset("spam")
    pipeline = build_pipeline(data.spec)
    X = data.X.iloc[:4].copy()
    X["message"] = ["hello friend", "secret bonus", "meeting friend", "claim bonus"]
    pipeline.fit(X, ["ham", "spam", "ham", "spam"])
    pipeline.predict(data.X.iloc[:1].assign(message="heldoutuniquetoken"))
    vocabulary = pipeline.named_steps["preprocess"].named_transformers_["text"].vocabulary_
    assert "heldoutuniquetoken" not in vocabulary


def test_text_holdout_supports_multistep_pipeline():
    result = train(load_dataset("spam"))
    assert result.report["metrics"]["accuracy"] >= 0.5
