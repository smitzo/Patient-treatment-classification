import pytest
from classification_lab.probabilities import validate_probabilities


@pytest.mark.parametrize("matrix", [[[0.5, 0.6]], [[-0.1, 1.1]], [[float("nan"), 0.5]], [[1]], []])
def test_invalid_probability_matrices_rejected(matrix):
    with pytest.raises(ValueError):
        validate_probabilities(matrix, ["a", "b"])
