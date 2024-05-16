import csv
from pathlib import Path


def test_spam_corpus_contains_unique_authored_rows():
    path = Path("src/classification_lab/resources/spam.csv")
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 128
    assert len({r["message"] for r in rows}) == 128
    assert {r["label"] for r in rows} == {"spam", "ham"}
