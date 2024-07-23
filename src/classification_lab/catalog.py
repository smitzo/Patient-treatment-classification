"""Offline provenance catalog shipped in the wheel."""

import json
from importlib.resources import files


def dataset_catalog():
    return json.loads(
        (files("classification_lab") / "resources" / "catalog.json").read_text(encoding="utf-8")
    )
