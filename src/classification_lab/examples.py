"""Common runner used by small standalone project scripts."""

import argparse
import json
from .datasets import load_dataset
from .experiments import train
from .reporting import write_json


def run_example(dataset, argv=None):
    parser = argparse.ArgumentParser(description=f"Educational {dataset} classification")
    parser.add_argument("--model", default="logistic", choices=["logistic", "forest", "dummy"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    parser.add_argument("--data-path")
    args = parser.parse_args(argv)
    kwargs = {"path": args.data_path} if args.data_path else {}
    result = train(load_dataset(dataset, **kwargs), args.model, args.seed)
    if args.output:
        write_json(args.output, result.report)
    print(json.dumps(result.report, allow_nan=False))
    return result
