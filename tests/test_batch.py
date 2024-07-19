import json
from classification_lab.batch import run_batch


def test_batch_runs_all_eleven_datasets_without_network(tmp_path):
    manifest = run_batch(output_dir=tmp_path)
    assert len(manifest["runs"]) == 11
    for item in manifest["runs"]:
        report = json.loads(__import__("pathlib").Path(item["path"]).read_text())
        assert report["reproducibility"]["dataset_fingerprint"]
    assert (tmp_path / "batch.json").exists()
