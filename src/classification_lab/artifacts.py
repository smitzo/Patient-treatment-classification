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


def load_artifact(path, *, trusted=False):
    if not trusted:
        raise ValueError("Refusing pickle input; explicitly trust this artifact before loading")
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or not {"metadata", "estimator"} <= payload.keys():
        raise ValueError("Malformed model artifact")
    metadata, estimator = payload["metadata"], payload["estimator"]
    if (
        metadata.get("format_version") != 1
        or metadata.get("sklearn_version") != sklearn.__version__
    ):
        raise ValueError("Unsupported artifact format or scikit-learn version")
    raw = dict(metadata["dataset_spec"])
    raw["numeric"], raw["categorical"] = tuple(raw["numeric"]), tuple(raw["categorical"])
    spec = DatasetSpec(**raw)
    if not hasattr(estimator, "predict_proba") or metadata["classes"] != [
        str(c) for c in estimator.classes_
    ]:
        raise ValueError("Artifact estimator does not match declared classes")
    return ModelArtifact(estimator, spec, metadata)
