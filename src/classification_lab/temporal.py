"""Chronological splits never place the same timestamp on both sides."""

import math
import pandas as pd
from .quality import validate_bundle
from .splitting import Split


def temporal_holdout(data, timestamps, test_size=0.25):
    validate_bundle(data)
    times = pd.to_datetime(pd.Series(list(timestamps)), errors="raise", utc=True)
    if len(times) != len(data.frame) or times.isna().any() or not 0 < test_size < 1:
        raise ValueError("Need aligned timestamps and test_size in (0,1)")
    ordered = times.sort_values(kind="stable")
    position = max(1, min(len(times) - 1, math.floor(len(times) * (1 - test_size))))
    boundary = ordered.iloc[position]
    fit = (times < boundary).to_numpy()
    test = ~fit
    if not fit.any() or not test.any() or data.y.iloc[fit].nunique() < 2:
        raise ValueError("Chronological training partition must contain at least two classes")
    return Split(data.X.iloc[fit], data.X.iloc[test], data.y.iloc[fit], data.y.iloc[test])
