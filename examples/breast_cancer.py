"""Inspect malignant-class errors in a bundled educational benchmark."""

import json
from classification_lab.projects import cancer_detection_study


if __name__ == "__main__":
    print(json.dumps(cancer_detection_study(), indent=2))
