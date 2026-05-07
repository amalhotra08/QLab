from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .quantum import QuantumStateEncoder, fidelity_kernel


@dataclass
class ModelOutput:
    logits: torch.Tensor
    attention: torch.Tensor


class ClassicalSelfAttention(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, attention_noise_std: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if attention_noise_std > 0:
            scores = scores + torch.randn_like(scores) * attention_noise_std
        attention = masked_softmax(scores, mask)
        return self.out(torch.matmul(self.dropout(attention), v)), attention


class ProjectedClassicalAttention(nn.Module):
    def __init__(self, dim: int, n_qubits: int, dropout: float):
        super().__init__()
        self.q_low = nn.Linear(dim, n_qubits)
        self.k_low = nn.Linear(dim, n_qubits)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, attention_noise_std: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_low(x)
        k = self.k_low(x)
        v = self.v(x)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if attention_noise_std > 0:
            scores = scores + torch.randn_like(scores) * attention_noise_std
        attention = masked_softmax(scores, mask)
        return self.out(torch.matmul(self.dropout(attention), v)), attention


class QuantumKernelAttention(nn.Module):
    def __init__(self, dim: int, n_qubits: int, depth: int, dropout: float):
        super().__init__()
        self.q_low = nn.Linear(dim, n_qubits)
        self.k_low = nn.Linear(dim, n_qubits)
        self.encoder = QuantumStateEncoder(n_qubits=n_qubits, depth=depth)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        self.angle_noise_std = 0.0
        self.depolarizing_prob = 0.0

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, attention_noise_std: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        q_state = self.encoder(self.q_low(x), angle_noise_std=self.angle_noise_std)
        k_state = self.encoder(self.k_low(x), angle_noise_std=self.angle_noise_std)
        overlap = fidelity_kernel(q_state, k_state, depolarizing_prob=self.depolarizing_prob)
        centered = overlap - (1.0 / q_state.shape[-1])
        scores = centered * torch.exp(self.logit_scale)
        if attention_noise_std > 0:
            scores = scores + torch.randn_like(scores) * attention_noise_std
        attention = masked_softmax(scores, mask)
        return self.out(torch.matmul(self.dropout(attention), self.v(x))), attention


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is not None:
        key_mask = mask.unsqueeze(1)
        scores = scores.masked_fill(~key_mask, -1e9)
    attention = torch.softmax(scores, dim=-1)
    if mask is not None:
        query_mask = mask.unsqueeze(-1).to(attention.dtype)
        attention = attention * query_mask
    return attention


class TinyAttentionClassifier(nn.Module):
    def __init__(
        self,
        model_type: str,
        vocab_size: int,
        max_len: int,
        num_classes: int = 4,
        dim: int = 48,
        hidden_dim: int = 96,
        n_qubits: int = 4,
        quantum_depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_type = model_type
        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, dim)
        if model_type == "classical":
            self.attention = ClassicalSelfAttention(dim, dropout)
        elif model_type == "classical_ablation":
            self.attention = ProjectedClassicalAttention(dim, n_qubits, dropout)
        elif model_type == "hybrid_quantum":
            self.attention = QuantumKernelAttention(dim, n_qubits, quantum_depth, dropout)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> ModelOutput:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        attended, attention = self.attention(x, attention_mask)
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ffn(x))
        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            weights = attention_mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return ModelOutput(logits=self.classifier(pooled), attention=attention)


def build_model(model_type: str, vocab_size: int, max_len: int, config) -> TinyAttentionClassifier:
    return TinyAttentionClassifier(
        model_type=model_type,
        vocab_size=vocab_size,
        max_len=max_len,
        dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        n_qubits=config.n_qubits,
        quantum_depth=config.quantum_depth,
        dropout=config.dropout,
    )

