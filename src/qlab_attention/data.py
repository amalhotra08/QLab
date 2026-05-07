from __future__ import annotations

import re
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .config import LABEL_NAMES, PROCESSED_DIR, ExperimentConfig
from .utils import write_json

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class SimpleTokenizer:
    vocab: dict[str, int]
    max_len: int
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @classmethod
    def build(cls, texts: list[str], vocab_size: int, max_len: int) -> "SimpleTokenizer":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))
        vocab = {"<pad>": 0, "<unk>": 1}
        for token, _ in counts.most_common(max(vocab_size - len(vocab), 0)):
            if token not in vocab:
                vocab[token] = len(vocab)
        return cls(vocab=vocab, max_len=max_len)

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokenize(text)]
        ids = ids[: self.max_len]
        mask = [1] * len(ids)
        if len(ids) < self.max_len:
            pad_count = self.max_len - len(ids)
            ids.extend([self.vocab[self.pad_token]] * pad_count)
            mask.extend([0] * pad_count)
        return ids, mask

    def save(self, path: Path) -> None:
        write_json(path, {"max_len": self.max_len, "vocab": self.vocab})


class TextClassificationDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: SimpleTokenizer):
        self.texts = texts
        self.labels = labels
        encoded = [tokenizer.encode(text) for text in texts]
        self.input_ids = torch.tensor([row[0] for row in encoded], dtype=torch.long)
        self.attention_mask = torch.tensor([row[1] for row in encoded], dtype=torch.bool)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.y[idx],
        }


def _records_from_huggingface(config: ExperimentConfig) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    from datasets import load_dataset

    dataset = load_dataset("ag_news", cache_dir=str(PROCESSED_DIR / "hf_cache"))
    needed_train = min(config.train_size + config.val_size, len(dataset["train"]))
    needed_test = min(config.test_size, len(dataset["test"]))
    shuffled_train = dataset["train"].shuffle(seed=config.seed).select(range(needed_train))
    shuffled_test = dataset["test"].shuffle(seed=config.seed).select(range(needed_test))
    train_records = [{"text": row["text"], "label": int(row["label"])} for row in shuffled_train]
    test_records = [{"text": row["text"], "label": int(row["label"])} for row in shuffled_test]
    return train_records, test_records, "huggingface_ag_news"


def _fallback_records(config: ExperimentConfig, error: Exception) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    templates = {
        0: [
            "Diplomats met to discuss the election and international security policy.",
            "The prime minister announced a regional peace agreement after negotiations.",
            "Global leaders debated trade rules during the summit in Geneva.",
        ],
        1: [
            "The basketball team won the championship after a late fourth quarter run.",
            "A record goal helped the club secure a playoff victory.",
            "The tennis final ended after five sets of dramatic rallies.",
        ],
        2: [
            "Technology stocks rose after the company reported stronger quarterly earnings.",
            "The central bank decision changed expectations for business lending.",
            "The airline announced a merger that reshaped the travel market.",
        ],
        3: [
            "Scientists released a new satellite image showing changes in ocean temperature.",
            "A software company introduced an artificial intelligence chip for mobile devices.",
            "Researchers tested a quantum processor with improved error rates.",
        ],
    }
    records: list[dict[str, object]] = []
    total = max(config.train_size + config.val_size + config.test_size, 96)
    for idx in range(total):
        label = idx % 4
        stem = templates[label][idx % len(templates[label])]
        records.append({"text": f"{stem} Report number {idx} adds local details.", "label": label})
    split = min(config.train_size + config.val_size, int(total * 0.75))
    reason = f"fallback_sample_used: {type(error).__name__}: {error}"
    return records[:split], records[split:], reason


def load_ag_news_records(config: ExperimentConfig) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    if os.environ.get("QLAB_FORCE_FALLBACK_DATA") == "1":
        return _fallback_records(config, RuntimeError("QLAB_FORCE_FALLBACK_DATA=1"))
    try:
        return _records_from_huggingface(config)
    except Exception as exc:  # pragma: no cover - exercised only when network/data is unavailable
        return _fallback_records(config, exc)


def make_dataloaders(config: ExperimentConfig) -> tuple[dict[str, DataLoader], SimpleTokenizer, dict[str, object]]:
    train_val_records, test_records, source = load_ag_news_records(config)
    train_records = train_val_records[: config.train_size]
    val_records = train_val_records[config.train_size : config.train_size + config.val_size]
    if not val_records:
        val_records = train_records[-max(4, len(train_records) // 5) :]
        train_records = train_records[: -len(val_records)]
    test_records = test_records[: config.test_size]

    train_texts = [str(row["text"]) for row in train_records]
    tokenizer = SimpleTokenizer.build(train_texts, config.vocab_size, config.max_len)

    datasets = {
        "train": TextClassificationDataset(train_texts, [int(row["label"]) for row in train_records], tokenizer),
        "val": TextClassificationDataset(
            [str(row["text"]) for row in val_records],
            [int(row["label"]) for row in val_records],
            tokenizer,
        ),
        "test": TextClassificationDataset(
            [str(row["text"]) for row in test_records],
            [int(row["label"]) for row in test_records],
            tokenizer,
        ),
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            generator=torch.Generator().manual_seed(config.seed),
        )
        for split, ds in datasets.items()
    }
    metadata = {
        "source": source,
        "label_names": LABEL_NAMES,
        "train_size": len(datasets["train"]),
        "val_size": len(datasets["val"]),
        "test_size": len(datasets["test"]),
        "vocab_size": len(tokenizer.vocab),
        "max_len": config.max_len,
        "seed": config.seed,
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(PROCESSED_DIR / "vocab.json")
    write_json(PROCESSED_DIR / "dataset_metadata.json", metadata)
    return loaders, tokenizer, metadata
