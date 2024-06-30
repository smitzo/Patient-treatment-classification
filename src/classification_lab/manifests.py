"""Deterministic run metadata without wall-clock noise or raw records."""

import hashlib
import importlib.metadata
import platform
import pandas as pd
from .fingerprints import dataset_fingerprint


def index_fingerprint(index):
    return hashlib.sha256(
        pd.util.hash_pandas_object(index, index=False).to_numpy(dtype="<u8").tobytes()
    ).hexdigest()


def reproducibility_manifest(data, split, seed):
    return {
        "dataset_fingerprint": dataset_fingerprint(data),
        "seed": seed,
        "source": data.spec.source,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")
        },
        "train_index_fingerprint": index_fingerprint(split.X_train.index),
        "test_index_fingerprint": index_fingerprint(split.X_test.index),
    }
