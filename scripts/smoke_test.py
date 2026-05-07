#!/usr/bin/env python
from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QLAB_FORCE_FALLBACK_DATA", "1")

from qlab_attention.config import ExperimentConfig, ensure_project_dirs
from qlab_attention.experiments import run_attention_alignment, run_gradient_variance, run_noise_sweep, run_sequence_benchmark
from qlab_attention.plots import make_all_figures
from qlab_attention.reporting import write_abstract
from qlab_attention.train import run_training_suite


def main() -> None:
    ensure_project_dirs()
    config = ExperimentConfig(train_size=48, val_size=16, test_size=16, epochs=1, batch_size=8, vocab_size=800, max_len=16, embedding_dim=24, hidden_dim=48)
    run_training_suite(config)
    run_sequence_benchmark(config, sequence_lengths=[8, 16], repeats=2)
    run_attention_alignment(config)
    run_gradient_variance(config, depths=[1, 2], seeds=[1, 2])
    run_noise_sweep(config, noise_levels=[0.0, 0.2])
    make_all_figures()
    write_abstract()
    print("smoke test completed")


if __name__ == "__main__":
    main()
