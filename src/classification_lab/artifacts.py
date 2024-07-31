"""Pickle is executable input: only load files you explicitly trust."""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import pickle
import platform
import tempfile
import sklearn
from .data import DatasetSpec
from .reporting import write_json


@dataclass
class ModelArtifact:
    estimator: object
    spec: DatasetSpec
    metadata: dict


def save_artifact(experiment, spec, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "dataset_spec": asdict(spec),
        "classes": [str(c) for c in experiment.estimator.classes_],
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "report": experiment.report,
    }
    validate_estimator(experiment.estimator, spec, metadata["classes"])
    payload = {"metadata": metadata, "estimator": experiment.estimator}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    write_json(str(path) + ".json", metadata)
    return path


def validate_estimator(estimator, spec, classes):
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or not all(isinstance(c, str) for c in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("Artifact needs distinct declared string classes")
    try:
        actual_classes = [str(c) for c in estimator.classes_]
        actual_features = list(estimator.feature_names_in_)
    except AttributeError as error:
        raise ValueError("Artifact needs a fitted estimator with named features") from error
    if (
        not callable(getattr(estimator, "predict_proba", None))
        or actual_classes != classes
        or actual_features != list(spec.features)
    ):
        raise ValueError("Artifact estimator does not match declared classes or feature schema")


def load_artifact(path, *, trusted=False):
    if not trusted:
        raise ValueError("Refusing pickle input; explicitly trust this artifact before loading")
    try:
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
    except (pickle.UnpicklingError, EOFError) as error:
        raise ValueError("Unreadable trusted model artifact") from error
    if not isinstance(payload, dict) or not {"metadata", "estimator"} <= payload.keys():
        raise ValueError("Malformed model artifact")
    metadata, estimator = payload["metadata"], payload["estimator"]
    if (
        not isinstance(metadata, dict)
        or metadata.get("format_version") != 1
        or metadata.get("sklearn_version") != sklearn.__version__
    ):
        raise ValueError("Unsupported artifact format or scikit-learn version")
    try:
        raw = dict(metadata["dataset_spec"])
        raw["numeric"], raw["categorical"] = tuple(raw["numeric"]), tuple(raw["categorical"])
        spec = DatasetSpec(**raw)
        validate_estimator(estimator, spec, metadata["classes"])
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed artifact schema metadata") from error
    return ModelArtifact(estimator, spec, metadata)
