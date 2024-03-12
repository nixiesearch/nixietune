from nixietune.crossencoder.trainer import CrossEncoderDataset
from transformers import AutoTokenizer
from datasets import Dataset
from nixietune.format.json import JSONDataset
from torch.utils.data.dataloader import DataLoader


def test_process_dataset():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    fmt = CrossEncoderDataset(tokenizer, max_len=128)
    data = Dataset.from_list([{"query": "hello", "doc": "cruel world", "neg": ["no"]}])
    parsed = JSONDataset.from_dataset(ds=data)
    processed = fmt.prepare_dataset(parsed, num_negatives=1).to_dict()
    assert processed == {
        "input_ids": [[101, 7592, 102, 10311, 2088, 102], [101, 7592, 102, 2053, 102]],
        "labels": [1.0, 0.0],
        "token_type_ids": [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1]],
        "attention_mask": [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
    }


def test_collate():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    fmt = CrossEncoderDataset(tokenizer, max_len=128)
    data = Dataset.from_list([{"query": "hello", "doc": "cruel world", "neg": ["no"]}])
    parsed = JSONDataset.from_dataset(ds=data)
    processed = fmt.prepare_dataset(parsed, num_negatives=1)
    loader = DataLoader(processed, collate_fn=fmt.collate, batch_size=2)
    batch = next(iter(loader))
    assert batch["input_ids"].tolist() == [
        [101, 7592, 102, 10311, 2088, 102, 0, 0],
        [101, 7592, 102, 2053, 102, 0, 0, 0],
    ]
    assert batch["token_type_ids"].tolist() == [
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
    ]
    assert batch["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
    ]
    assert batch["labels"].tolist() == [1.0, 0.0]


def test_collate_bge():
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
    fmt = CrossEncoderDataset(tokenizer, max_len=128)
    data = Dataset.from_list([{"query": "hello", "doc": "cruel world", "neg": ["no"]}])
    parsed = JSONDataset.from_dataset(ds=data)
    processed = fmt.prepare_dataset(parsed, num_negatives=1)
    loader = DataLoader(processed, collate_fn=fmt.collate, batch_size=2)
    batch = next(iter(loader))
    assert batch["input_ids"].tolist() == [
        [0, 33600, 31, 2, 2, 156217, 8999, 2],
        [0, 33600, 31, 2, 2, 110, 2, 1],
    ]
    assert batch["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0],
    ]
    assert batch["labels"].tolist() == [1.0, 0.0]
