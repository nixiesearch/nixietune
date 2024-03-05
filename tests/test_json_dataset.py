from datasets import Dataset
from nixietune.format.json import JSONDataset
from transformers import AutoTokenizer
import pytest


def test_query_pos():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = Dataset.from_list([{"query": "foo", "pos": "bar"}])
    parsed = JSONDataset.from_dataset(ds=ds, tokenizer=tokenizer, max_len=128).to_dict()
    assert parsed == {
        "query": [[101, 29379, 102]],
        "query_text": ["foo"],
        "pos": [[101, 3347, 102]],
        "pos_text": ["bar"],
        "neg": [[]],
        "negscore": [[]],
    }


def test_query_pos_neg():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = Dataset.from_list([{"query": "foo", "pos": "bar", "neg": ["bar"]}])
    parsed = JSONDataset.from_dataset(ds=ds, tokenizer=tokenizer, max_len=128).to_dict()
    assert parsed == {
        "query": [[101, 29379, 102]],
        "query_text": ["foo"],
        "pos": [[101, 3347, 102]],
        "pos_text": ["bar"],
        "neg": [[[101, 3347, 102]]],
        "negscore": [[0]],
    }


def test_query_pos_neg2():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = Dataset.from_list([{"query": "foo", "pos": "bar", "neg": ["bar", "bar"]}])
    parsed = JSONDataset.from_dataset(ds=ds, tokenizer=tokenizer, max_len=128).to_dict()
    assert parsed == {
        "query": [[101, 29379, 102]],
        "query_text": ["foo"],
        "pos": [[101, 3347, 102]],
        "pos_text": ["bar"],
        "neg": [[[101, 3347, 102], [101, 3347, 102]]],
        "negscore": [[0, 0]],
    }


def test_query_pos_empty_neg():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = Dataset.from_list([{"query": "foo", "pos": "bar", "neg": []}])
    parsed = JSONDataset.from_dataset(ds=ds, tokenizer=tokenizer, max_len=128).to_dict()
    assert parsed == {
        "query": [[101, 29379, 102]],
        "query_text": ["foo"],
        "pos": [[101, 3347, 102]],
        "pos_text": ["bar"],
        "neg": [[]],
        "negscore": [[]],
    }


def test_query_pos_neg_scores():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = Dataset.from_list([{"query": "foo", "pos": "bar", "neg": ["bar", "bar"], "negscore": [0.5, 0.5]}])
    parsed = JSONDataset.from_dataset(ds=ds, tokenizer=tokenizer, max_len=128).to_dict()
    assert parsed == {
        "query": [[101, 29379, 102]],
        "query_text": ["foo"],
        "pos": [[101, 3347, 102]],
        "pos_text": ["bar"],
        "neg": [[[101, 3347, 102], [101, 3347, 102]]],
        "negscore": [[0.5, 0.5]],
    }


def test_fail_negscore_mismatch():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = Dataset.from_list([{"query": "foo", "pos": "bar", "neg": ["bar", "bar"], "negscore": [0.5]}])
    with pytest.raises(Exception) as error:
        parsed = JSONDataset.from_dataset(ds=ds, tokenizer=tokenizer, max_len=128).to_dict()


def test_only_pos():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = Dataset.from_list([{"pos": "bar"}])
    parsed = JSONDataset.from_dataset(ds=ds, tokenizer=tokenizer, max_len=128).to_dict()
    assert parsed == {
        "query": [None],
        "query_text": [None],
        "pos": [[101, 3347, 102]],
        "pos_text": ["bar"],
        "neg": [[]],
        "negscore": [[]],
    }
