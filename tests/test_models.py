import torch

from qlab_attention.config import ExperimentConfig
from qlab_attention.models import build_model


def test_model_forward_shapes_and_attention_rows():
    config = ExperimentConfig(vocab_size=100, max_len=8, embedding_dim=16, hidden_dim=32, n_qubits=4, quantum_depth=1)
    input_ids = torch.randint(2, 100, (3, 8))
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    for model_type in ["classical", "classical_ablation", "hybrid_quantum"]:
        model = build_model(model_type, vocab_size=100, max_len=8, config=config)
        output = model(input_ids, mask)
        assert output.logits.shape == (3, 4)
        assert output.attention.shape == (3, 8, 8)
        row_sums = output.attention.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_padding_query_rows_are_zero():
    config = ExperimentConfig(vocab_size=100, max_len=6, embedding_dim=16, hidden_dim=32)
    input_ids = torch.randint(2, 100, (2, 6))
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    model = build_model("classical", vocab_size=100, max_len=6, config=config)
    output = model(input_ids, mask)
    assert torch.all(output.attention[0, 3:] == 0)

