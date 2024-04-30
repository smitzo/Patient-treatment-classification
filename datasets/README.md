# Dataset sources

The original `patient_data.csv` contains 4,412 records and is preserved byte for byte.
`SOURCE` describes recorded in/out care status. It does not establish a treatment recommendation.
The original README attributes it to [Patient Treatment Classification on Kaggle](https://www.kaggle.com/datasets/saurabhshahane/patient-treatment-classification).
The collection procedure, consent, measurement units, population representativeness and redistribution rights have not been independently verified.
Do not infer these from this repository. No new real patient data is added.

`patient.schema.json` records the columns actually observed in the existing file. Numeric missing
values are permitted for training-fold imputation; targets cannot be missing. Recorded SEX values
are F/M; these describe the source schema and do not establish completeness of demographic coverage.
