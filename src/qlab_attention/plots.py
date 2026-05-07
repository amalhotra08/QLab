from __future__ import annotations

import os
from pathlib import Path

_MPL_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[2] / ".cache"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, METRICS_DIR

PALETTE = {
    "classical": "#234f73",
    "classical_ablation": "#6f7f52",
    "hybrid_quantum": "#8f3f5f",
}


def _setup() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"figure.dpi": 160, "savefig.bbox": "tight"})


def plot_summary() -> Path | None:
    path = METRICS_DIR / "summary.csv"
    if not path.exists():
        return None
    _setup()
    df = pd.read_csv(path)
    melted = df.melt(id_vars=["model_type"], value_vars=["test_accuracy", "test_macro_f1"], var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.barplot(data=melted, x="model_type", y="score", hue="metric", palette=["#234f73", "#b87745"], ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(1.0, melted["score"].max() + 0.08))
    ax.set_title("AG News Classification Performance")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="")
    out = FIGURES_DIR / "accuracy_f1_summary.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_benchmark() -> Path | None:
    path = METRICS_DIR / "benchmark_seq_len.csv"
    if not path.exists():
        return None
    _setup()
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.lineplot(data=df, x="sequence_length", y="mean_forward_ms", hue="model_type", marker="o", palette=PALETTE, ax=ax)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Mean forward time (ms)")
    ax.set_title("Inference Runtime Scaling")
    out = FIGURES_DIR / "benchmark_runtime.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_noise() -> Path | None:
    path = METRICS_DIR / "noise_sweep.csv"
    if not path.exists():
        return None
    _setup()
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.lineplot(data=df, x="noise_level", y="test_accuracy", hue="model_type", marker="o", palette=PALETTE, ax=ax)
    ax.set_xlabel("Simulated quantum noise level")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Noise Robustness")
    out = FIGURES_DIR / "noise_robustness.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_gradient() -> Path | None:
    path = METRICS_DIR / "gradient_variance.csv"
    if not path.exists():
        return None
    _setup()
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.lineplot(data=df, x="depth", y="grad_variance", marker="o", color="#8f3f5f", ax=ax)
    ax.set_xlabel("Quantum circuit depth")
    ax.set_ylabel("Mean gradient variance")
    ax.set_title("Trainability Diagnostic")
    out = FIGURES_DIR / "gradient_variance.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_alignment() -> Path | None:
    path = METRICS_DIR / "attention_alignment.csv"
    if not path.exists():
        return None
    _setup()
    df = pd.read_csv(path)
    df["pair"] = df["left_model"] + " vs " + df["right_model"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.barplot(data=df, x="pair", y="linear_cka", color="#4f7a8f", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Linear CKA")
    ax.set_ylim(0, 1.0)
    ax.set_title("Attention Map Alignment")
    ax.tick_params(axis="x", rotation=20)
    out = FIGURES_DIR / "attention_alignment.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_architecture() -> Path:
    _setup()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.axis("off")
    boxes = [
        (0.05, 0.62, 0.18, 0.16, "Token\nEmbeddings"),
        (0.30, 0.70, 0.18, 0.12, "Query\nProjection"),
        (0.30, 0.50, 0.18, 0.12, "Key\nProjection"),
        (0.55, 0.60, 0.22, 0.18, "4-Qubit Simulated\nPQC Kernel\n|<psi(q)|psi(k)>|^2"),
        (0.30, 0.22, 0.18, 0.12, "Value\nProjection"),
        (0.82, 0.58, 0.14, 0.16, "Softmax\nWeights"),
        (0.68, 0.22, 0.18, 0.12, "Weighted\nValues"),
        (0.82, 0.20, 0.14, 0.12, "Classifier"),
    ]
    for x, y, w, h, label in boxes:
        color = "#efe8dd" if "PQC" in label else "#d9e7ec"
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#27333a", linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)
    arrows = [
        ((0.23, 0.70), (0.30, 0.76)),
        ((0.23, 0.70), (0.30, 0.56)),
        ((0.48, 0.76), (0.55, 0.69)),
        ((0.48, 0.56), (0.55, 0.65)),
        ((0.77, 0.69), (0.82, 0.66)),
        ((0.23, 0.70), (0.30, 0.28)),
        ((0.48, 0.28), (0.68, 0.28)),
        ((0.89, 0.58), (0.77, 0.34)),
        ((0.86, 0.28), (0.86, 0.32)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#27333a", lw=1.5))
    ax.text(0.05, 0.90, "Hybrid Quantum-Classical Attention Block", fontsize=18, fontweight="bold", ha="left")
    out = FIGURES_DIR / "architecture_diagram.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def make_all_figures() -> list[Path]:
    outputs = [plot_architecture()]
    for fn in [plot_summary, plot_benchmark, plot_noise, plot_gradient, plot_alignment]:
        result = fn()
        if result is not None:
            outputs.append(result)
    return outputs
