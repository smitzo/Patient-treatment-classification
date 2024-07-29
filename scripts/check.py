"""Run the repository's repeatable local checks with the active interpreter."""

from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    commands = [
        [sys.executable, "-m", "ruff", "check", "src", "tests", "examples", "scripts"],
        [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "examples", "scripts"],
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "-m", "pip", "check"],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
