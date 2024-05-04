"""Show the effect of training-only standardization on chemical measurements."""

import json
from classification_lab.projects import wine_scaling_study


if __name__ == "__main__":
    print(json.dumps(wine_scaling_study(), indent=2))
