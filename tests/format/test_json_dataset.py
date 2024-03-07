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
