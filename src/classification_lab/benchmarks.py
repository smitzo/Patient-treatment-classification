"""Adapters for small datasets distributed with scikit-learn."""

from sklearn.datasets import load_iris
from .data import DatasetBundle, DatasetSpec
from .quality import validate_bundle


def from_bunch(name, bunch):
    frame = bunch.data.copy()
    frame.columns = frame.columns.astype(str)
    frame["label"] = bunch.target.map(lambda value: str(bunch.target_names[int(value)]))
    spec = DatasetSpec(
        name,
        "label",
        numeric=tuple(bunch.feature_names),
        source=f"scikit-learn bundled {name}; see upstream dataset description",
    )
    bundle = DatasetBundle(frame, spec)
    validate_bundle(bundle)
    return bundle


def iris():
    return from_bunch("iris", load_iris(as_frame=True))
