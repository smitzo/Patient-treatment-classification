"""Compare heldout errors for synthetic pumps, compressors and motors."""

import json
from classification_lab.projects import equipment_type_study


if __name__ == "__main__":
    print(json.dumps(equipment_type_study(), indent=2))
