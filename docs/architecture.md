# Architecture

The package separates dataset contracts from estimators and output formats. `DatasetSpec` defines
ordered numeric, categorical or text predictors and a distinct target. `DatasetBundle` exposes
copies of features and labels, so feature edits do not mutate the source frame.

Data flows from a registered offline loader through schema validation, a split boundary, a fitted
scikit-learn pipeline, metrics and a versioned JSON report. The patient loader reads the original
external CSV. Small benchmarks come from the installed scikit-learn distribution. Synthetic and
authored text CSVs are wheel resources. The provenance catalog distinguishes these origins.

Numeric pipelines impute medians and standardize; categorical pipelines impute common values and
one-hot encode unseen categories without error. Text pipelines learn a bounded TF-IDF vocabulary.
Each pipeline owns its transformations. Cross-validation and grid search clone the whole pipeline,
so imputation, scaling, encoding and vocabulary learning remain inside each training fold.

`train` owns a single stratified holdout. `tune` reserves that holdout before training-only model
selection. Calibration and threshold studies have explicit nested training boundaries. Model
comparison and feature ablation are selection diagnostics; callers must reserve evaluation data
before using them to choose a final model.

An `Experiment` holds a fitted estimator, a report and in-memory partitions. JSON output excludes
raw feature records. Reproducibility manifests contain package versions, source, seed and content
and index fingerprints. Dataset content hashes are versioned with the pandas hashing scheme;
they are evidence of content identity within that environment, not anonymization or authenticity.

CLI commands register argument parsers and invoke the same Python functions. Commands emit JSON
to stdout, with explicit paths for reports, CSVs and model artifacts. A batch run writes one report
per project and a manifest. Partial outputs can exist if a later project fails; a manifest is only
written after the whole batch succeeds.

Static HTML escapes labels and JSON text. Matplotlib creates noninteractive figures and clears
them after saving. Reports and artifacts are local ignored outputs. Pickle models are executable
inputs and require explicit trust; the JSON inspection command never opens the model file.
