from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"
PAPER_DIR = PROJECT_ROOT / "paper"
POSTER_DIR = PROJECT_ROOT / "poster"

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


@dataclass
class ExperimentConfig:
    seed: int = 42
    train_size: int = 1200
    val_size: int = 300
    test_size: int = 300
    vocab_size: int = 8000
    max_len: int = 32
    batch_size: int = 32
    epochs: int = 2
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    embedding_dim: int = 48
    hidden_dim: int = 96
    dropout: float = 0.1
    n_qubits: int = 4
    quantum_depth: int = 2
    device: str = "cpu"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def ensure_project_dirs() -> None:
    for path in [
        DATA_DIR,
        PROCESSED_DIR,
        RESULTS_DIR,
        METRICS_DIR,
        CHECKPOINT_DIR,
        FIGURES_DIR,
        DOCS_DIR,
        PAPER_DIR,
        POSTER_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

