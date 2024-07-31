"""Read CSVs without implicit indexes, ambiguous headers or truncated rows."""

import csv
from pathlib import Path
import pandas as pd
from .data import DatasetBundle, DatasetSpec


def read_checked_csv(path, dtype=None):
    path = Path(path)
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, [])
            if not header or any(not column.strip() for column in header):
                raise ValueError("CSV needs nonempty column names")
            if len(header) != len(set(header)):
                raise ValueError("Duplicate CSV column names")
            for row in reader:
                if row and len(row) != len(header):
                    raise ValueError(
                        f"CSV row ending at line {reader.line_num} has {len(row)} fields; expected {len(header)}"
                    )
    except csv.Error as error:
        raise ValueError(f"Malformed CSV: {error}") from error
    return pd.read_csv(path, dtype=dtype, encoding="utf-8-sig")


def load_csv(path: str | Path, spec: DatasetSpec) -> DatasetBundle:
    # Preserve numeric-looking labels and categories, including leading zeroes.
    dtype = {column: str for column in spec.categorical + (spec.target,)}
    if spec.text:
        dtype[spec.text] = str
    frame = read_checked_csv(path, dtype=dtype)
    expected = set(spec.features + (spec.target,))
    if set(frame.columns) != expected:
        raise ValueError(f"Expected columns {sorted(expected)}, received {list(frame.columns)}")
    if frame.empty or frame[spec.target].isna().any():
        raise ValueError("Dataset must contain rows and nonmissing targets")
    if frame[spec.target].nunique() < 2:
        raise ValueError("Classification requires at least two target classes")
    return DatasetBundle(frame, spec)
