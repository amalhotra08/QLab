from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from .config import CHECKPOINT_DIR, METRICS_DIR, ExperimentConfig
from .data import SimpleTokenizer, make_dataloaders
from .metrics import classification_metrics
from .models import TinyAttentionClassifier, build_model
from .utils import choose_device, set_seed, write_json

MODEL_TYPES = ["classical", "classical_ablation", "hybrid_quantum"]


def train_epoch(model: TinyAttentionClassifier, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for batch in tqdm(loader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, mask)
        loss = criterion(output.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * labels.numel()
        total += labels.numel()
    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate(model: TinyAttentionClassifier, loader, criterion, device: torch.device) -> dict[str, float]:
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    total_loss = 0.0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        output = model(input_ids, mask)
        loss = criterion(output.logits, labels)
        pred = output.logits.argmax(dim=-1)
        all_true.extend(labels.detach().cpu().tolist())
        all_pred.extend(pred.detach().cpu().tolist())
        total_loss += float(loss.detach().cpu()) * labels.numel()
        total += labels.numel()
    metrics = classification_metrics(all_true, all_pred)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def save_checkpoint(path: Path, model: TinyAttentionClassifier, tokenizer: SimpleTokenizer, config: ExperimentConfig, model_type: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "config": config.to_dict(),
            "vocab_size": len(tokenizer.vocab),
            "max_len": tokenizer.max_len,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> TinyAttentionClassifier:
    payload = torch.load(path, map_location=device)
    config = ExperimentConfig(**payload["config"])
    model = build_model(payload["model_type"], payload["vocab_size"], payload["max_len"], config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def train_model(
    model_type: str,
    config: ExperimentConfig,
    loaders,
    tokenizer: SimpleTokenizer,
) -> dict[str, float | str | int]:
    set_seed(config.seed)
    device = choose_device(config.device)
    model = build_model(model_type, len(tokenizer.vocab), config.max_len, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    start = time.perf_counter()
    history: list[dict[str, float | int | str]] = []
    best_val = -1.0
    best_state = None
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        history.append(
            {
                "model_type": model_type,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], criterion, device)
    elapsed = time.perf_counter() - start
    checkpoint_path = CHECKPOINT_DIR / f"{model_type}.pt"
    save_checkpoint(checkpoint_path, model, tokenizer, config, model_type)
    pd.DataFrame(history).to_csv(METRICS_DIR / f"history_{model_type}.csv", index=False)
    row: dict[str, float | str | int] = {
        "model_type": model_type,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_loss": test_metrics["loss"],
        "best_val_accuracy": best_val,
        "train_seconds": elapsed,
        "epochs": config.epochs,
        "checkpoint": str(checkpoint_path),
    }
    return row


def run_training_suite(config: ExperimentConfig, model_types: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    loaders, tokenizer, metadata = make_dataloaders(config)
    rows = [train_model(model_type, config, loaders, tokenizer) for model_type in (model_types or MODEL_TYPES)]
    summary = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(METRICS_DIR / "summary.csv", index=False)
    write_json(METRICS_DIR / "run_config.json", {"config": config.to_dict(), "dataset": metadata})
    return summary, metadata

