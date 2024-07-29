"""Install a wheel into a temporary target and run outside the checkout.

Dependencies come from the active interpreter; the package must come from the wheel.
No index access is permitted during this smoke check.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel")
    parser.add_argument("--patient-path")
    args = parser.parse_args(argv)
    wheel = Path(args.wheel).resolve()
    patient = str(Path(args.patient_path).resolve()) if args.patient_path else None
    if not wheel.is_file():
        parser.error("Wheel does not exist; build it first")
    with tempfile.TemporaryDirectory(prefix="classification-wheel-") as temporary:
        target = Path(temporary) / "installed"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--target",
                str(target),
                str(wheel),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        env = {
            **os.environ,
            "PYTHONPATH": str(target),
            "PYTHONNOUSERSITE": "1",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(Path(temporary) / "mpl"),
        }
        probe = """
import json, pathlib, sys
import classification_lab
from classification_lab.catalog import dataset_catalog
from classification_lab.datasets import load_dataset
from classification_lab.experiments import train
assert pathlib.Path(classification_lab.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]))
items = dataset_catalog()["datasets"]
for item in items:
    if item["name"] != "patient":
        assert len(load_dataset(item["name"]).frame) == item["rows"]
for name in ["iris", "spam"]:
    assert train(load_dataset(name)).report["test_rows"] > 0
if sys.argv[2]:
    assert len(load_dataset("patient",path=sys.argv[2]).frame) == 4412
print(json.dumps({"wheel_datasets":len(items),"package_path":classification_lab.__file__}))
"""
        output = subprocess.run(
            [sys.executable, "-c", probe, str(target), patient or ""],
            cwd=temporary,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        executable = target / "bin" / "classification-lab"
        command = subprocess.run(
            [str(executable), "datasets"],
            cwd=temporary,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        assert len(json.loads(command.stdout)["datasets"]) == 11
        print(output.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
