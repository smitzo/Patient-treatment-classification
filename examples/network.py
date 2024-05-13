"""Inspect protocol support and errors on synthetic connection records."""

import json
from classification_lab.projects import network_protocol_study


if __name__ == "__main__":
    print(json.dumps(network_protocol_study(), indent=2))
