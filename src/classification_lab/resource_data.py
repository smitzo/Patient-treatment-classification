"""Strict loaders for versioned educational CSV resources."""

from importlib.resources import files
from .csv_io import load_csv
from .data import DatasetSpec
from .quality import validate_bundle


RESOURCE_SPECS = {}


def load_resource(name):
    if name not in RESOURCE_SPECS:
        raise ValueError(f"Unknown resource {name!r}")
    bundle = load_csv(
        files("classification_lab") / "resources" / f"{name}.csv", RESOURCE_SPECS[name]
    )
    validate_bundle(bundle)
    return bundle


RESOURCE_SPECS["churn"] = DatasetSpec(
    "churn",
    "label",
    ("tenure_months", "monthly_cost", "support_calls"),
    ("contract",),
    source="Synthetic logistic generator; seed 42; 600 rows",
)

RESOURCE_SPECS["equipment"] = DatasetSpec(
    "equipment",
    "label",
    ("vibration", "temperature", "load_percent", "operating_hours"),
    ("machine_type",),
    source="Synthetic failure generator; seed 42; 600 rows",
)

RESOURCE_SPECS["transactions"] = DatasetSpec(
    "transactions",
    "label",
    ("amount", "hour", "recent_count"),
    ("foreign",),
    source="Synthetic transaction generator; seed 42; 600 rows; not real financial data",
)
