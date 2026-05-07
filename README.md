# QLab Hybrid Quantum Attention

This workspace contains a compact, reproducible implementation of the QLab senior research project:

**A Hybrid Quantum-Classical Attention Mechanism for Efficient Large Language Models**

The project compares three matched AG News text classifiers:

- `classical`: standard scaled dot-product self-attention.
- `hybrid_quantum`: Q/K projections are encoded into a 4-qubit simulated quantum circuit; attention logits come from Hilbert-space state overlap.
- `classical_ablation`: uses the same low-dimensional Q/K bottleneck as the hybrid model, but computes similarity classically.

The code writes real metrics to `results/metrics/`, figures to `figures/`, and generated writing artifacts to `docs/`, `paper/`, and `poster/`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Quick Checks

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/smoke_test.py
```

## Full Compact Run

```bash
.venv/bin/python scripts/run_all.py --train-size 1200 --val-size 300 --test-size 300 --epochs 2
```

This downloads AG News from Hugging Face when network access is available. If the download fails, the pipeline falls back to a tiny bundled sample and records that limitation in `data/processed/dataset_metadata.json`.

## Main Outputs

- `results/metrics/summary.csv`
- `results/metrics/benchmark_seq_len.csv`
- `results/metrics/noise_sweep.csv`
- `results/metrics/gradient_variance.csv`
- `results/metrics/attention_alignment.csv`
- `figures/*.png`
- `docs/abstract.md`
- `paper/final_paper.md`
- `paper/final_paper.pdf`
- `poster/QLab_Hybrid_Quantum_Attention_Poster.pptx`
- `poster/QLab_Hybrid_Quantum_Attention_Poster.pdf`

