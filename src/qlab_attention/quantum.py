from __future__ import annotations

import math

import torch
from torch import nn


class QuantumStateEncoder(nn.Module):
    """Differentiable state-vector simulation for a small hardware-efficient circuit.

    The circuit starts in |0000>, re-uploads the projected token features through
    RY rotations, and applies nearest-neighbor CZ gates after each upload layer.
    Similarity is measured as fidelity, |<psi(q)|psi(k)>|^2.
    """

    def __init__(self, n_qubits: int = 4, depth: int = 2):
        super().__init__()
        if n_qubits < 1 or n_qubits > 8:
            raise ValueError("This compact simulator expects 1-8 qubits.")
        self.n_qubits = n_qubits
        self.depth = depth
        self.n_states = 2**n_qubits
        self.theta = nn.Parameter(0.05 * torch.randn(depth, n_qubits))
        self.input_scale = nn.Parameter(torch.ones(n_qubits))
        self.register_buffer("basis", torch.arange(self.n_states, dtype=torch.long), persistent=False)
        for qubit in range(n_qubits):
            idx0 = [idx for idx in range(self.n_states) if ((idx >> qubit) & 1) == 0]
            idx1 = [idx | (1 << qubit) for idx in idx0]
            self.register_buffer(f"idx0_{qubit}", torch.tensor(idx0, dtype=torch.long), persistent=False)
            self.register_buffer(f"idx1_{qubit}", torch.tensor(idx1, dtype=torch.long), persistent=False)
        phase = torch.ones(self.n_states)
        for left in range(n_qubits - 1):
            both_one = (((self.basis >> left) & 1) == 1) & (((self.basis >> (left + 1)) & 1) == 1)
            phase = torch.where(both_one, -phase, phase)
        self.register_buffer("cz_phase", phase, persistent=False)

    def _apply_ry(self, state: torch.Tensor, angle: torch.Tensor, qubit: int) -> torch.Tensor:
        idx0 = getattr(self, f"idx0_{qubit}")
        idx1 = getattr(self, f"idx1_{qubit}")
        original0 = state.index_select(-1, idx0)
        original1 = state.index_select(-1, idx1)
        c = torch.cos(angle / 2.0).unsqueeze(-1)
        s = torch.sin(angle / 2.0).unsqueeze(-1)
        rotated = state.clone()
        rotated[..., idx0] = c * original0 - s * original1
        rotated[..., idx1] = s * original0 + c * original1
        return rotated

    def forward(self, projected_features: torch.Tensor, angle_noise_std: float = 0.0) -> torch.Tensor:
        if projected_features.shape[-1] != self.n_qubits:
            raise ValueError("Last dimension must equal n_qubits.")
        features = projected_features
        if angle_noise_std > 0:
            features = features + torch.randn_like(features) * angle_noise_std

        state = torch.zeros(*features.shape[:-1], self.n_states, device=features.device, dtype=features.dtype)
        state[..., 0] = 1.0
        bounded = torch.tanh(features) * math.pi
        for layer in range(self.depth):
            for qubit in range(self.n_qubits):
                angle = bounded[..., qubit] * self.input_scale[qubit] + self.theta[layer, qubit]
                state = self._apply_ry(state, angle, qubit)
            state = state * self.cz_phase.to(device=state.device, dtype=state.dtype)
        return torch.nn.functional.normalize(state, p=2, dim=-1)


def fidelity_kernel(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    depolarizing_prob: float = 0.0,
) -> torch.Tensor:
    overlap = torch.matmul(query_states, key_states.transpose(-1, -2)).pow(2)
    if depolarizing_prob > 0:
        uniform = 1.0 / query_states.shape[-1]
        overlap = (1.0 - depolarizing_prob) * overlap + depolarizing_prob * uniform
    return overlap.clamp(0.0, 1.0)

