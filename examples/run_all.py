"""Run all project datasets or pass selected names and batch options."""

import sys
from classification_lab.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["batch", *sys.argv[1:]]))
