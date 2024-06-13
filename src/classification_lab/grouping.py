"""Keep every entity or template group on one side of a split."""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .quality import validate_bundle
from .splitting import Split


def group_holdout(data, groups, test_size=0.25, seed=42):
    validate_bundle(data)
    groups = pd.Series(list(groups))
    if len(groups) != len(data.frame) or groups.isna().any() or groups.nunique() < 2:
        raise ValueError("Need aligned nonmissing groups with at least two distinct values")
    if not 0 < test_size < 1:
        raise ValueError("test_size must lie in (0,1)")
    splitter = GroupShuffleSplit(n_splits=30, test_size=test_size, random_state=seed)
    for fit, test in splitter.split(data.X, data.y, groups):
        if set(data.y.iloc[fit]) == set(data.y) and set(data.y.iloc[test]) == set(data.y):
            return Split(data.X.iloc[fit], data.X.iloc[test], data.y.iloc[fit], data.y.iloc[test])
    raise ValueError("Cannot find a group-disjoint partition containing every class")
