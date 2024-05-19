import csv
from pathlib import Path


def test_spam_corpus_contains_unique_authored_rows():
    path = Path("src/classification_lab/resources/spam.csv")
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 128
    assert len({r["message"] for r in rows}) == 128
    assert {r["label"] for r in rows} == {"spam", "ham"}


def test_text_bundle_excludes_label_and_rejects_blank_text():
    import pytest
    from classification_lab.datasets import load_dataset
    from classification_lab.quality import validate_bundle

    data = load_dataset("spam")
    assert data.X.columns.tolist() == ["message"]
    data.frame.loc[0, "message"] = "   "
    with pytest.raises(ValueError, match="nonempty"):
        validate_bundle(data)


def test_sentiment_corpus_has_both_balanced_classes():
    from classification_lab.datasets import load_dataset

    data = load_dataset("sentiment")
    assert data.y.value_counts().to_dict() == {"positive": 64, "negative": 64}
    assert not data.frame.message.duplicated().any()
