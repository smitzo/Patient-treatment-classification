# Artifact contract and trust

A saved model is a Python pickle containing a fitted estimator and a metadata dictionary. Format
version 1 records its dataset specification, class order, Python/scikit-learn versions and source
experiment report. The estimator preserves fitted preprocessing, feature order and model state.
It is trained on the experiment's training partition, not silently refitted on the full dataset.

Pickle deserialization can execute arbitrary code before any format or version validation runs.
The loader refuses all inputs unless the caller explicitly sets `trusted=True`; the CLI requires
`--trust-artifact`. These flags express a trust decision. They do not sandbox, authenticate, scan
or make an artifact safe. Obtain artifacts only through a trusted local workflow.

After deserialization, the loader checks the format and exact scikit-learn version, reconstructs
the dataset specification and compares declared classes with estimator classes. Use matching
Python and dependency environments; this format is not a stable cross-version interchange
protocol. Compatibility checks cannot undo code that ran during unpickling.

The companion `.pkl.json` file contains JSON metadata only. Inspection reads this sidecar without
opening the pickle, and explicitly identifies it as unverified. The sidecar is editable and is
not a signature, authenticated digest or proof of correspondence to a model. Missing or invalid
sidecars fail inspection; model loading uses embedded metadata.

Model bytes are written to a private temporary file and atomically replaced. JSON reports use the
same replace pattern and reject nonfinite values before touching the destination. The model and
sidecar are separate writes; a failure can leave a model without its matching sidecar. There is no
multi-file transaction or guarantee of durable directory fsync. Explicit output paths may replace
existing outputs and should be chosen intentionally.

Prediction requires exactly the schema's feature columns, with nonempty rows and compatible value
types. Column order is restored, unknown categories are accepted by the fitted encoder, numeric
missing values use fitted imputers, and targets/extra columns are rejected. Outputs preserve row
order, expose named class probabilities, and validate probability shape, bounds and row sums.

JSON and HTML reports exclude raw input rows but can include dataset names, labels, source paths,
feature names and learned text vocabulary in project studies. Pickle artifacts expose model
internals. Local ignored output is not a privacy, retention or production security policy.
