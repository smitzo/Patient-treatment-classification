"""Summarize mistaken digit pairs from 8-by-8 pixel classification."""

import json
from classification_lab.projects import digit_error_study


if __name__ == "__main__":
    print(json.dumps(digit_error_study(), indent=2))
