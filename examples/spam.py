"""Inspect indicative TF-IDF terms learned from authored toy messages."""

import json
from classification_lab.projects import text_term_study


if __name__ == "__main__":
    print(json.dumps(text_term_study("spam"), indent=2))
