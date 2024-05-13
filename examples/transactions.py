"""Evaluate rare synthetic anomalies with explicit positive-class metrics."""

import json
from classification_lab.projects import transaction_detection_study


if __name__ == "__main__":
    print(json.dumps(transaction_detection_study(), indent=2))
