# Classification Lab

A Python workbench for 11 educational classification projects, starting with the original Patient
Treatment Classification notebook. Train tabular and text pipelines, compare models, tune on
training folds, inspect errors, export reports and predict with trusted local models.

The original `patient_data.csv` and `patient_treatment_classification.ipynb` are preserved.
`classification_lab.ipynb` is a reproducible companion with no stored performance claims.

## Install

Use Python 3.10 or newer. The locked development environment was resolved and tested locally on
CPython 3.10/Linux; automated checks are configured for 3.10 and 3.12. Hosted CI has not been run
as part of local development.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps --no-build-isolation -e .
```

For a regular installation with dependency ranges, use `python -m pip install -r requirements.txt`.
Notebook users can install Jupyter separately and select this environment's kernel.
Once installed, dataset loading and experiments need no network access.

## Projects and datasets

| Project | Dataset origin | What the example examines |
| --- | --- | --- |
| Patient treatment | Existing 4,412-record CSV | Recorded in/out care-source prediction |
| Iris | Bundled benchmark | Sepal versus petal measurements |
| Wine | Bundled benchmark | Effect of feature scaling |
| Breast cancer | Bundled benchmark | Explicit malignant-class sensitivity |
| Digits | Bundled benchmark | Frequent digit-confusion pairs |
| Customer churn | Synthetic | Errors by contract type |
| Equipment failure | Synthetic | Errors by machine type |
| Transaction anomaly | Synthetic | Minority-class precision, recall and AP |
| Network intrusion | Synthetic | Errors by protocol |
| Spam | Authored toy messages | Indicative TF-IDF terms |
| Sentiment | Authored toy reviews | Review polarity and terms |

The four synthetic tabular datasets include reproducible generators and CSV snapshots. Two text
corpora contain authored templates. New CSVs and the provenance catalog live in
`src/classification_lab/resources/`; benchmark data comes from scikit-learn's local bundle.
See [dataset provenance](docs/data-provenance.md) and [patient metadata](datasets/README.md).
Patient redistribution rights and collection details remain unverified. Original synthetic and
authored example data is CC0-1.0; third-party data is not relicensed.

## Train and inspect

Run these commands from the repository root after activating the environment:

```bash
classification-lab datasets
classification-lab validate patient
classification-lab train patient --model logistic --seed 42 --output reports/patient.json
classification-lab train iris --model forest --output reports/iris-forest.json
classification-lab train spam --output reports/spam.json
classification-lab compare wine --models dummy logistic forest --output reports/wine-cv.json
classification-lab tune iris --model logistic --output reports/iris-tuned.json
classification-lab report reports/patient.json --html reports/patient.html --confusion-plot reports/patient.png
```

`python -m classification_lab` exposes the same interface. `--data-path /path/to/patient_data.csv`
selects a patient CSV when working outside this directory. Model choices are `dummy`, `logistic`
and `forest`. Metrics include accuracy, balanced accuracy, macro F1, per-class values and an
explicitly ordered confusion matrix. Seeds, package versions, content and split fingerprints
make the evaluation boundary inspectable. Generated outputs stay ignored by Git.

## Run project studies

```bash
python examples/patient.py --model forest --output reports/patient-forest.json
python examples/iris.py
python examples/wine.py
python examples/breast_cancer.py
python examples/digits.py
python examples/churn.py
python examples/equipment.py
python examples/transactions.py
python examples/network.py
python examples/spam.py
python examples/sentiment.py
python examples/run_all.py --output-dir reports/all
```

The batch runner evaluates every registered dataset. Select specific projects with
`python examples/run_all.py iris spam --output-dir reports/selected`.
Study scripts print JSON; the patient example and batch runner accept the options shown above.

## Save a model and predict

```bash
classification-lab train iris --artifact artifacts/iris.pkl --output reports/iris.json
classification-lab inspect artifacts/iris.pkl
python -c 'from classification_lab.datasets import load_dataset; load_dataset("iris").X.head().to_csv("artifacts/iris-input.csv", index=False)'
classification-lab predict artifacts/iris.pkl artifacts/iris-input.csv --trust-artifact --output-csv artifacts/iris-predictions.csv
```

Only supply `--trust-artifact` for a model from a trusted source. Pickle loading can execute code;
format checks cannot make an untrusted model safe. Inspection reads an unverified JSON sidecar
without opening the model. Predictions require feature columns only, preserve row order and
include a named probability column for every class. See [artifact contracts](docs/artifacts.md).

## Generate and evaluate other records

```bash
classification-lab generate churn --rows 1000 --seed 9 --output-csv artifacts/churn-1000.csv
classification-lab train patient --model dummy --output reports/patient-dummy.json
classification-lab leaderboard reports/patient.json reports/patient-dummy.json
```

Generated CSVs are teaching material and can be loaded with the Python CSV adapter and a matching
`DatasetSpec`; the named training commands use the versioned dataset snapshots. The `evaluate`
command scores a CSV containing aligned `truth` and `prediction` columns:
`classification-lab evaluate labels.csv --classes in out`. Supply the complete class order.
Leaderboards require matching dataset and heldout-population fingerprints.

## Python evaluation tools

```python
from classification_lab.datasets import load_dataset
from classification_lab.experiments import train
from classification_lab.importance import heldout_importance
from classification_lab.curves import binary_curves
from classification_lab.thresholds import train_thresholded

result = train(load_dataset("churn"), seed=42)
importance = heldout_importance(result)
curves = binary_curves(result.split.y_test,
                       result.estimator.predict_proba(result.split.X_test),
                       result.estimator.classes_, positive="churn")
threshold_study = train_thresholded(load_dataset("churn"), positive="churn")
```

Additional modules provide training-fold calibration, learning curves, feature ablation,
group/chronological splits, subgroup support and paired bootstrap intervals.
[Evaluation contracts](docs/evaluation.md) explain their assumptions and limits;
[architecture](docs/architecture.md) explains how the components fit together.

## Validate and build

```bash
python scripts/check.py
python -m build --wheel --no-isolation
python scripts/smoke_wheel.py dist/classification_lab-0.1.0-py3-none-any.whl --patient-path patient_data.csv
```

The wheel check installs into a temporary target, imports outside the checkout, checks packaged
datasets and invokes the console command. It reuses the active interpreter's dependencies.
The GitHub Actions workflow runs linting, tests and distribution checks when pushed.

These projects are educational experiments. Synthetic correlations and template-corpus accuracy
do not establish real-world utility. Patient and breast-cancer benchmarks do not establish
clinical validity, and local tests do not establish deployment acceptance. The historical
notebook's recorded accuracy is not a current validated result.
