"""Reject score comparisons across datasets or holdout populations."""

import json
from pathlib import Path


def leaderboard(paths, metric="macro_f1"):
    if metric not in {"accuracy", "balanced_accuracy", "macro_f1"} or not paths:
        raise ValueError("Choose reports and a supported classification metric")
    reports = [json.loads(Path(path).read_text()) for path in paths]
    signatures = set()
    for report in reports:
        manifest = report.get("reproducibility", {})
        required = (manifest.get("dataset_fingerprint"), manifest.get("test_index_fingerprint"))
        if not all(required):
            raise ValueError("Reports need dataset and test-partition fingerprints")
        signatures.add(required)
    if len(signatures) != 1:
        raise ValueError("Leaderboard reports must share dataset and heldout population")
    rows = [
        {"model": r["model"], "seed": r["seed"], "score": r["metrics"][metric], "path": str(p)}
        for p, r in zip(paths, reports)
    ]
    rows.sort(key=lambda row: (-row["score"], row["model"]))
    return {
        "dataset": reports[0]["dataset"],
        "metric": metric,
        "rows": rows,
        "scope": "descriptive shared-holdout comparison; repeated selection biases this score",
    }
