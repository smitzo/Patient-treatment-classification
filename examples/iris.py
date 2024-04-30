"""Compare sepal and petal measurements on identical seeded iris splits."""

import json
from classification_lab.projects import iris_feature_study


if __name__ == "__main__":
    print(json.dumps(iris_feature_study(), indent=2))
