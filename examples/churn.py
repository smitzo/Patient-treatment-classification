"""Toy churn errors by contract; synthetic data has no real customer records."""

import json
from classification_lab.projects import churn_contract_study


if __name__ == "__main__":
    print(json.dumps(churn_contract_study(), indent=2))
