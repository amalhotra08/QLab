#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlab_attention.config import ExperimentConfig, ensure_project_dirs
from qlab_attention.train import run_training_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    ensure_project_dirs()
    config = ExperimentConfig(train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, epochs=args.epochs)
    summary, _ = run_training_suite(config)
    print(summary)


if __name__ == "__main__":
    main()

