"""Descriptive subgroup metrics; small groups do not establish fairness."""

import numpy as np
import pandas as pd
from .metrics import classification_metrics


def subgroup_metrics(y_true, y_pred, groups, classes, min_support=5):
    if len(y_true) != len(y_pred) or len(y_true) != len(groups) or min_support < 1:
        raise ValueError("Truth, prediction and groups must align; min_support must be positive")
    group_values = pd.Series(list(groups), dtype="object").fillna("<missing>").astype(str)
    truth, predicted = np.asarray(y_true), np.asarray(y_pred)
    reports = {}
    for group in sorted(group_values.unique()):
        mask = (group_values == group).to_numpy()
        support = int(mask.sum())
        reports[group] = {
            "support": support,
            "suppressed": support < min_support,
            "metrics": classification_metrics(truth[mask], predicted[mask], classes)
            if support >= min_support
            else None,
        }
    return reports
