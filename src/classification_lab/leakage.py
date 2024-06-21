"""Exact feature fingerprints flag repeated records independently of labels."""

import pandas as pd


def feature_hashes(frame):
    return pd.util.hash_pandas_object(frame, index=False)


def split_overlap(X_train, X_test):
    if X_train.columns.tolist() != X_test.columns.tolist():
        raise ValueError("Split feature schemas must agree")
    train_hashes, test_hashes = feature_hashes(X_train), feature_hashes(X_test)
    overlap = set(train_hashes).intersection(test_hashes)
    return {
        "shared_feature_patterns": len(overlap),
        "affected_train_rows": int(train_hashes.isin(overlap).sum()),
        "affected_test_rows": int(test_hashes.isin(overlap).sum()),
        "method": "64-bit pandas feature hashes; investigate matches, not identity proof",
    }
