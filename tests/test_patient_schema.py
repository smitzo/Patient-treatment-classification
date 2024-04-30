import json
from pathlib import Path
from classification_lab.datasets import load_patient


def test_published_schema_matches_preserved_csv():
    schema = json.loads(Path("datasets/patient.schema.json").read_text())
    data = load_patient()
    assert schema["rows"] == len(data.frame)
    assert schema["numeric_features"] == list(data.spec.numeric)
    assert schema["target"]["labels"] == sorted(data.y.unique())
