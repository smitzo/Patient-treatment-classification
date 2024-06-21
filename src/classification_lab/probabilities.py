"""Probability contract shared by prediction and evaluation."""

import numpy as np


def validate_probabilities(probabilities, classes, rows=None):
    classes = list(classes)
    matrix = np.asarray(probabilities, dtype=float)
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Probability columns need distinct class labels")
    if matrix.ndim != 2 or matrix.shape[1] != len(classes) or not len(matrix):
        raise ValueError("Expected a nonempty rows-by-classes probability matrix")
    if rows is not None and matrix.shape[0] != rows:
        raise ValueError("Probability rows do not align")
    if not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any():
        raise ValueError("Probabilities must be finite and within [0,1]")
    if not np.allclose(matrix.sum(axis=1), 1, atol=1e-7, rtol=0):
        raise ValueError("Probability rows must sum to one")
    return matrix
