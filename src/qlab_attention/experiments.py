from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn

from .config import CHECKPOINT_DIR, METRICS_DIR, ExperimentConfig
from .data import make_dataloaders
from .metrics import linear_cka
from .models import QuantumKernelAttention, build_model
from .train import MODEL_TYPES, evaluate, load_checkpoint
from .utils import choose_device, set_seed


def _checkpoint_or_fresh(model_type: str, vocab_size: int, config: ExperimentConfig, device: torch.device):
    path = CHECKPOINT_DIR / f"{model_type}.pt"
    if path.exists():
        return load_checkpoint(path, device)
    model = build_model(model_type, vocab_size, config.max_len, config).to(device)
    model.eval()
    return model


@torch.no_grad()
def run_sequence_benchmark(config: ExperimentConfig, sequence_lengths: list[int] | None = None, repeats: int = 12) -> pd.DataFrame:
    set_seed(config.seed)
    device = choose_device(config.device)
    sequence_lengths = sequence_lengths or [8, 16, 32, 64]
    max_len = max(sequence_lengths)
    bench_config = ExperimentConfig(**{**config.to_dict(), "max_len": max_len})
    vocab_size = max(config.vocab_size, 1000)
    process = psutil.Process()
    rows: list[dict[str, float | int | str]] = []
    for model_type in MODEL_TYPES:
        model = build_model(model_type, vocab_size, max_len, bench_config).to(device)
        model.eval()
        for seq_len in sequence_lengths:
            input_ids = torch.randint(2, vocab_size, (config.batch_size, seq_len), device=device)
            mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
            for _ in range(3):
                _ = model(input_ids, mask)
            timings = []
            rss_before = process.memory_info().rss / (1024 * 1024)
            for _ in range(repeats):
                start = time.perf_counter()
                _ = model(input_ids, mask)
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000)
            rss_after = process.memory_info().rss / (1024 * 1024)
            rows.append(
                {
                    "model_type": model_type,
                    "sequence_length": seq_len,
                    "batch_size": config.batch_size,
                    "mean_forward_ms": float(np.mean(timings)),
                    "std_forward_ms": float(np.std(timings)),
                    "rss_delta_mb": float(max(rss_after - rss_before, 0.0)),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "benchmark_seq_len.csv", index=False)
    return df


@torch.no_grad()
def run_attention_alignment(config: ExperimentConfig) -> pd.DataFrame:
    loaders, tokenizer, _ = make_dataloaders(config)
    device = choose_device(config.device)
    batch = next(iter(loaders["test"]))
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    attention_maps: dict[str, torch.Tensor] = {}
    for model_type in MODEL_TYPES:
        model = _checkpoint_or_fresh(model_type, len(tokenizer.vocab), config, device)
        output = model(input_ids, mask)
        flat = output.attention.detach().cpu().reshape(-1, output.attention.shape[-1])
        valid_rows = flat.abs().sum(dim=1) > 0
        attention_maps[model_type] = flat[valid_rows]
    rows = []
    pairs = [("classical", "hybrid_quantum"), ("classical", "classical_ablation"), ("classical_ablation", "hybrid_quantum")]
    for left, right in pairs:
        x = attention_maps[left]
        y = attention_maps[right]
        n = min(len(x), len(y))
        rows.append(
            {
                "left_model": left,
                "right_model": right,
                "linear_cka": linear_cka(x[:n], y[:n]),
                "mean_abs_attention_diff": float((x[:n] - y[:n]).abs().mean()),
                "samples": int(n),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "attention_alignment.csv", index=False)
    return df


def _quantum_gradient_stats(model, batch, device: torch.device) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.train()
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    model.zero_grad(set_to_none=True)
    loss = criterion(model(input_ids, mask).logits, labels)
    loss.backward()
    grads = []
    for name, param in model.named_parameters():
        if "attention.encoder" in name and param.grad is not None:
            grads.append(param.grad.detach().flatten().cpu())
    if not grads:
        return {"grad_mean_abs": 0.0, "grad_variance": 0.0, "grad_norm": 0.0}
    vector = torch.cat(grads)
    return {
        "grad_mean_abs": float(vector.abs().mean()),
        "grad_variance": float(vector.var(unbiased=False)),
        "grad_norm": float(vector.norm()),
    }


def run_gradient_variance(config: ExperimentConfig, depths: list[int] | None = None, seeds: list[int] | None = None) -> pd.DataFrame:
    depths = depths or [1, 2, 3, 4, 5]
    seeds = seeds or [11, 23, 42, 77, 101]
    loaders, tokenizer, _ = make_dataloaders(config)
    batch = next(iter(loaders["train"]))
    device = choose_device(config.device)
    rows = []
    for depth in depths:
        for seed in seeds:
            depth_config = ExperimentConfig(**{**config.to_dict(), "quantum_depth": depth, "seed": seed})
            set_seed(seed)
            model = build_model("hybrid_quantum", len(tokenizer.vocab), depth_config.max_len, depth_config).to(device)
            stats = _quantum_gradient_stats(model, batch, device)
            rows.append({"depth": depth, "seed": seed, **stats})
    df = pd.DataFrame(rows)
    aggregate = df.groupby("depth", as_index=False).agg(
        grad_mean_abs=("grad_mean_abs", "mean"),
        grad_variance=("grad_variance", "mean"),
        grad_norm=("grad_norm", "mean"),
    )
    df.to_csv(METRICS_DIR / "gradient_variance_raw.csv", index=False)
    aggregate.to_csv(METRICS_DIR / "gradient_variance.csv", index=False)
    return aggregate


def _set_noise(model, angle_noise_std: float, depolarizing_prob: float) -> None:
    for module in model.modules():
        if isinstance(module, QuantumKernelAttention):
            module.angle_noise_std = angle_noise_std
            module.depolarizing_prob = depolarizing_prob


def run_noise_sweep(config: ExperimentConfig, noise_levels: list[float] | None = None) -> pd.DataFrame:
    noise_levels = noise_levels or [0.0, 0.05, 0.10, 0.20, 0.30]
    loaders, tokenizer, _ = make_dataloaders(config)
    device = choose_device(config.device)
    criterion = nn.CrossEntropyLoss()
    rows = []
    for model_type in ["classical", "classical_ablation", "hybrid_quantum"]:
        model = _checkpoint_or_fresh(model_type, len(tokenizer.vocab), config, device)
        for noise in noise_levels:
            if model_type == "hybrid_quantum":
                _set_noise(model, angle_noise_std=noise, depolarizing_prob=min(noise, 0.9))
            metrics = evaluate(model, loaders["test"], criterion, device)
            rows.append(
                {
                    "model_type": model_type,
                    "noise_level": noise,
                    "test_accuracy": metrics["accuracy"],
                    "test_macro_f1": metrics["macro_f1"],
                    "note": "quantum angle + depolarizing noise" if model_type == "hybrid_quantum" else "no quantum module; baseline repeated",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "noise_sweep.csv", index=False)
    return df
