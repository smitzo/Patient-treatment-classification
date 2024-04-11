"""Read CSVs without silently accepting ambiguous schemas."""

import csv
from pathlib import Path
import pandas as pd
from .data import DatasetBundle, DatasetSpec


def load_csv(path: str | Path, spec: DatasetSpec) -> DatasetBundle:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    expected = set(spec.features + (spec.target,))
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names")
    if set(header) != expected:
        raise ValueError(f"Expected columns {sorted(expected)}, received {header}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty or frame[spec.target].isna().any():
        raise ValueError("Dataset must contain rows and nonmissing targets")
    if frame[spec.target].nunique() < 2:
        raise ValueError("Classification requires at least two target classes")
    return DatasetBundle(frame, spec)
