"""Permutation importance for raw input columns, including categorical predictors."""

from sklearn.inspection import permutation_importance


def heldout_importance(experiment, repeats=5, seed=42):
    if repeats < 1:
        raise ValueError("At least one permutation repeat is required")
    result = permutation_importance(
        experiment.estimator,
        experiment.split.X_test,
        experiment.split.y_test,
        n_repeats=repeats,
        random_state=seed,
        scoring="balanced_accuracy",
        n_jobs=1,
    )
    rows = [
        {"feature": str(feature), "mean": float(mean), "std": float(std)}
        for feature, mean, std in zip(
            experiment.split.X_test.columns, result.importances_mean, result.importances_std
        )
    ]
    return sorted(rows, key=lambda row: (-row["mean"], row["feature"]))
