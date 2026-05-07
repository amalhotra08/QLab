#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlab_attention.config import ExperimentConfig, ensure_project_dirs
from qlab_attention.experiments import run_attention_alignment, run_gradient_variance, run_noise_sweep, run_sequence_benchmark
from qlab_attention.plots import make_all_figures
from qlab_attention.reporting import write_all_artifacts
from qlab_attention.train import run_training_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the compact QLab hybrid quantum attention project.")
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--benchmark-repeats", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    config = ExperimentConfig(
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        device=args.device,
    )
    print("1/6 training models")
    summary, metadata = run_training_suite(config)
    print(summary)
    print(f"dataset: {metadata}")
    print("2/6 benchmarking sequence lengths")
    run_sequence_benchmark(config, repeats=args.benchmark_repeats)
    print("3/6 analyzing attention alignment")
    run_attention_alignment(config)
    print("4/6 measuring quantum gradient variance")
    run_gradient_variance(config)
    print("5/6 sweeping simulated noise")
    run_noise_sweep(config)
    print("6/6 generating figures and writing artifacts")
    make_all_figures()
    outputs = write_all_artifacts()
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

