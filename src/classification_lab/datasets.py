"""Offline dataset registry. Patient data stays in the original repository CSV."""

from pathlib import Path
from .data import DatasetSpec
from .csv_io import load_csv
from .quality import validate_bundle

PATIENT = DatasetSpec(
    "patient",
    "SOURCE",
    numeric=(
        "HAEMATOCRIT",
        "HAEMOGLOBINS",
        "ERYTHROCYTE",
        "LEUCOCYTE",
        "THROMBOCYTE",
        "MCH",
        "MCHC",
        "MCV",
        "AGE",
    ),
    categorical=("SEX",),
    source="Existing patient_data.csv; upstream rights unverified",
)


def load_patient(path: str | Path = "patient_data.csv"):
    data = load_csv(path, PATIENT)
    if not set(data.y).issubset({"in", "out"}):
        raise ValueError("Patient SOURCE must be in or out")
    if not set(data.frame.SEX.dropna()).issubset({"F", "M"}):
        raise ValueError("Patient SEX must be F or M")
    validate_bundle(data)
    return data


LOADERS = {"patient": load_patient}


def load_dataset(name: str, **kwargs):
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(LOADERS)}")
    return LOADERS[name](**kwargs)
