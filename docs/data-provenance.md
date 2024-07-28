# Data provenance and limitations

The machine-readable `classification_lab/resources/catalog.json` is the source catalog. It records
origins, expected row counts and SHA-256 values for checked-in CSVs. The catalog does not grant
rights to third-party material. Changing data requires updating its generator and catalog together.

| Dataset | Records | Origin | Target |
| --- | ---: | --- | --- |
| patient | 4,412 | Existing repository CSV | Recorded in/out care source |
| iris | 150 | scikit-learn bundle | Three flower species |
| wine | 178 | scikit-learn bundle | Three wine cultivars |
| breast_cancer | 569 | scikit-learn bundle | Malignant/benign benchmark |
| digits | 1,797 | scikit-learn bundle | Ten handwritten digits |
| churn | 600 | Seeded synthetic generator | Churn/stay |
| equipment | 600 | Seeded synthetic generator | Failure/normal |
| transactions | 600 | Seeded synthetic generator | Anomaly/normal |
| network | 600 | Seeded synthetic generator | Intrusion/normal |
| spam | 128 | Authored templates | Spam/ham |
| sentiment | 128 | Authored templates | Positive/negative |

Patient metadata comes from the existing file and README attribution, not independent verification.
Collection methods, measurement units, consent, representativeness and redistribution rights remain
unknown. The old notebook is retained as historical work; its recorded outputs are not new results.

Bundled benchmarks are loaded locally with scikit-learn. Their installed loader descriptions and
`DESCR` fields retain upstream references and terms. The adapters name class labels and exclude
labels from features. No remote dataset is fetched at runtime. See the upstream
[dataset descriptions](https://scikit-learn.org/stable/datasets/toy_dataset.html) for source context.
No new third-party licensing assertion is made here.

Original tabular examples use NumPy's seeded generator and logistic probabilities sampled into
labels. Features intentionally contain overlap and noise. They contain no real customer, device,
transaction or network records; numeric scales are illustrative. Default snapshots use seed 42
and 600 records, and tests compare each CSV to its generator. Their artificial correlations are
not estimates of real-world relationships. Generic classification generation uses scikit-learn.

Text corpora combine 16 authored templates and 8 topics. They contain no real communications,
customer reviews, working spam links or personal contact details. Random splits share vocabulary
and templates; holdout accuracy is intentionally limited evidence. Neutral sentiment, sarcasm,
multilingual inputs and realistic adversarial variation are not covered.

The four original synthetic CSV datasets and two authored text CSV datasets are dedicated under
[CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/). This dedication applies only to those
new example datasets. Existing patient data, benchmark data and source-code licensing are separate.
