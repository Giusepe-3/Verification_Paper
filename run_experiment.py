#!/usr/bin/env python3
"""
run_experiment.py
-----------------
Entry point for the verification-collapse experiment.

Usage
-----
    python run_experiment.py                         # uses config.yaml
    python run_experiment.py --config config.yaml    # explicit path
    python run_experiment.py --iterations 3          # override num_iterations
"""

import argparse
import sys

from src.experiment import VerificationCollapseExperiment


def main() -> None:
    # Force line-buffered stdout so logs appear immediately when redirected to a file
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Run the verification-collapse experiment.")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to YAML config (default: config.yaml)"
    )
    parser.add_argument(
        "--iterations", type=int, default=None, help="Override num_iterations from config"
    )
    args = parser.parse_args()

    exp = VerificationCollapseExperiment.from_config(args.config)
    exp.run(num_iterations=args.iterations)


if __name__ == "__main__":
    main()
