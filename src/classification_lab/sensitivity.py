"""Repeated holdouts describe split sensitivity; observations are not independent."""

import numpy as np
from .experiments import train


def seed_sensitivity(data, seeds=(7, 19, 42), model="logistic"):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Choose at least two distinct seeds")
    reports = [train(data, model, seed=seed).report for seed in seeds]
    scores = [r["metrics"]["macro_f1"] for r in reports]
    return {
        "dataset": data.spec.name,
        "model": model,
        "seeds": list(seeds),
        "reports": reports,
        "macro_f1_mean": float(np.mean(scores)),
        "macro_f1_std": float(np.std(scores, ddof=1)),
        "scope": "overlapping repeated holdouts; not independent confidence samples",
    }
