from nixietune.format.json import JSONDataset
from datasets import Dataset


def test_query_only():
    ds = Dataset.from_list([{"query": "foo"}, {"query": "bar"}])
    result = JSONDataset.from_dataset(ds).to_dict()
    assert result == {
        "query": ["foo", "bar"],
        "doc": [None, None],
        "neg": [[], []],
        "negscore": [[], []],
    }


def test_doc_only():
    ds = Dataset.from_list([{"doc": "foo"}, {"doc": "bar"}])
    result = JSONDataset.from_dataset(ds).to_dict()
    assert result == {
        "doc": ["foo", "bar"],
        "query": [None, None],
        "neg": [[], []],
        "negscore": [[], []],
    }


def test_negs():
    ds = Dataset.from_list([{"query": "foo", "doc": "bar", "neg": ["no"], "negscore": [0]}])
    result = JSONDataset.from_dataset(ds).to_dict()
    assert result == {
        "doc": ["bar"],
        "query": ["foo"],
        "neg": [["no"]],
        "negscore": [[0]],
    }


def test_negs_no_score():
    ds = Dataset.from_list([{"query": "foo", "doc": "bar", "neg": ["no"]}])
    result = JSONDataset.from_dataset(ds).to_dict()
    assert result == {
        "doc": ["bar"],
        "query": ["foo"],
        "neg": [["no"]],
        "negscore": [[0]],
    }
