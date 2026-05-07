#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qlab_attention.config import ExperimentConfig, ensure_project_dirs
from qlab_attention.experiments import run_attention_alignment, run_gradient_variance, run_noise_sweep, run_sequence_benchmark
from qlab_attention.plots import make_all_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-repeats", type=int, default=12)
    args = parser.parse_args()
    ensure_project_dirs()
    config = ExperimentConfig()
    print(run_sequence_benchmark(config, repeats=args.benchmark_repeats))
    print(run_attention_alignment(config))
    print(run_gradient_variance(config))
    print(run_noise_sweep(config))
    make_all_figures()


if __name__ == "__main__":
    main()

