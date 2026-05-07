import torch

from qlab_attention.data import SimpleTokenizer, TextClassificationDataset, tokenize


def test_tokenize_and_encode_padding():
    assert tokenize("Hello, QLab's WORLD!") == ["hello", "qlab's", "world"]
    tokenizer = SimpleTokenizer.build(["hello world", "hello quantum"], vocab_size=10, max_len=5)
    ids, mask = tokenizer.encode("hello missing")
    assert len(ids) == 5
    assert len(mask) == 5
    assert mask == [1, 1, 0, 0, 0]
    assert ids[0] == tokenizer.vocab["hello"]
    assert ids[1] == tokenizer.vocab["<unk>"]


def test_dataset_shapes():
    tokenizer = SimpleTokenizer.build(["sports story", "business story"], vocab_size=10, max_len=4)
    dataset = TextClassificationDataset(["sports story", "business story"], [1, 2], tokenizer)
    row = dataset[0]
    assert row["input_ids"].shape[0] == 4
    assert row["attention_mask"].dtype == torch.bool
    assert int(row["labels"]) == 1
