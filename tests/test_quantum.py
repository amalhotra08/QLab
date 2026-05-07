import torch

from qlab_attention.quantum import QuantumStateEncoder, fidelity_kernel


def test_quantum_state_normalization():
    encoder = QuantumStateEncoder(n_qubits=4, depth=2)
    features = torch.randn(5, 3, 4)
    states = encoder(features)
    norms = states.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_fidelity_kernel_symmetry_and_range():
    encoder = QuantumStateEncoder(n_qubits=4, depth=1)
    features = torch.randn(2, 5, 4)
    states = encoder(features)
    kernel = fidelity_kernel(states, states)
    assert kernel.shape == (2, 5, 5)
    assert torch.allclose(kernel, kernel.transpose(-1, -2), atol=1e-5)
    assert float(kernel.detach().min()) >= 0.0
    assert float(kernel.detach().max()) <= 1.0
