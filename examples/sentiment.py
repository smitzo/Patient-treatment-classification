"""Run review polarity classification on an authored toy corpus."""

import json
from classification_lab.projects import text_term_study


if __name__ == "__main__":
    print(json.dumps(text_term_study("sentiment"), indent=2))
