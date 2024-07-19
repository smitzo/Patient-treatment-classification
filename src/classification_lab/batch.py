"""Repeatable batch runs with one report per dataset and a manifest."""

from pathlib import Path
from .datasets import LOADERS, load_dataset
from .experiments import train
from .reporting import write_json


def run_batch(
    names=None,
    output_dir="reports/batch",
    model="logistic",
    seed=42,
    patient_path="patient_data.csv",
):
    names = sorted(LOADERS) if names is None else list(names)
    if not names or len(set(names)) != len(names) or not set(names).issubset(LOADERS):
        raise ValueError("Choose distinct registered dataset names")
    output_dir = Path(output_dir)
    reports = []
    for name in names:
        kwargs = {"path": patient_path} if name == "patient" else {}
        result = train(load_dataset(name, **kwargs), model, seed)
        path = write_json(output_dir / f"{name}-{model}-{seed}.json", result.report)
        reports.append(
            {"dataset": name, "path": str(path), "macro_f1": result.report["metrics"]["macro_f1"]}
        )
    manifest = {
        "model": model,
        "seed": seed,
        "runs": reports,
        "scope": "offline educational holdouts",
    }
    write_json(output_dir / "batch.json", manifest)
    return manifest
