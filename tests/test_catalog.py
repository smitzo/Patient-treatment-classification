import hashlib
from pathlib import Path
from classification_lab.catalog import dataset_catalog
from classification_lab.datasets import LOADERS, load_dataset


def test_catalog_matches_registered_datasets_and_resource_bytes():
    catalog = dataset_catalog()
    assert {item["name"] for item in catalog["datasets"]} == set(LOADERS)
    for item in catalog["datasets"]:
        assert len(load_dataset(item["name"]).frame) == item["rows"]
        if "file_sha256" in item:
            path = (
                Path("patient_data.csv")
                if item["name"] == "patient"
                else Path(f"src/classification_lab/resources/{item['name']}.csv")
            )
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["file_sha256"]
