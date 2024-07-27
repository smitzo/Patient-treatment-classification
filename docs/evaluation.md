# Evaluation contracts

A stratified holdout preserves every observed target class on both sides. Split seeds repeat row
partitions for a fixed dataframe order and environment. Index fingerprints record row identity;
dataset fingerprints detect feature, target, order, schema and dependency changes.

Training-only grid search uses macro F1 and refits the winning pipeline on the reserved training
partition. Its best cross-validation score is a selection statistic. The single final holdout
score has a different meaning. Repeating selection on holdout outcomes makes that score optimistic.
A leaderboard requires matching content and test-partition fingerprints; this verifies population
comparability, not an unbiased model-selection procedure or equivalent fitting budgets.

Metrics keep explicit class order in confusion matrices and per-class summaries. Accuracy can hide
minority-class failure. Balanced accuracy averages recall over classes present in the evaluated
truth; macro F1 uses all declared classes. Undefined precision/recall values use zero. Probability
matrices must be finite, bounded, aligned with classes and sum to one. Binary ROC/PR diagnostics
require both classes and an explicit positive class; ROC infinity thresholds become JSON null.

Calibration fits only inside training folds. Threshold selection uses fit/validation/test
partitions: choose a threshold with validation F1, apply it once to the outer test partition,
and retain the original fitted model. Refitting afterward could change probability calibration.
The generic selection function cannot determine where caller-provided labels originated.

Group splits keep every entity or template on one side and require all classes in both partitions.
Chronological splits move entire equal-timestamp groups across a boundary, so the achieved fraction
can differ from the requested fraction. Training must contain two classes; future test classes can
be absent. Neither split invents missing patient/entity identifiers or timestamps.

Exact feature-hash overlap detects repeated patterns across a random holdout, regardless of label.
It is reported rather than silently removed: deduplication can alter population meaning. Hash
matches merit investigation. Use grouping by a genuine entity/template identifier where available.
No duplicate detection proves that other leakage is absent.

Bootstrap percentile intervals resample paired fixed predictions. They exclude retraining and
population shift uncertainty. Repeated seed runs overlap and are not independent samples.
Learning curves refit transforms at each training size; tiny fractions can be infeasible and fail
explicitly. Feature-ablation callers must provide training-only data. Heldout permutation
importance is descriptive, can be negative, and is not a causal feature effect.

Subgroup outputs include support and suppress metrics below a configurable count. Support does not
establish fairness or clinical validity. The tiny authored text corpora share templates across
random splits; their scores test pipeline behavior, not real-message generalization. None of these
local experiments establish clinical, security, fraud-detection or deployment acceptance.
