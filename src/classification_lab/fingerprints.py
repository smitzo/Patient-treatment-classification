"""Content identity includes schema, dtypes, row order and values, but not row index labels."""

from dataclasses import asdict
import hashlib
import json
import pandas as pd


def dataset_fingerprint(data):
    frame = data.frame.loc[:, list(data.spec.features) + [data.spec.target]]
    header = {
        "spec": asdict(data.spec),
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "hash_scheme": "pandas-row-hash-v1",
        "pandas_version": pd.__version__,
    }
    digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="<u8").tobytes())
    return digest.hexdigest()
