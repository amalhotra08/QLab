from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.detach().float()
    y = y.detach().float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    hsic = torch.linalg.norm(x.T @ y, ord="fro").pow(2)
    x_norm = torch.linalg.norm(x.T @ x, ord="fro")
    y_norm = torch.linalg.norm(y.T @ y, ord="fro")
    return float((hsic / (x_norm * y_norm + eps)).clamp(0.0, 1.0).cpu())

