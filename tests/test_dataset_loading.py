from datasets import load_dataset, Features, Value, Sequence
from nixietune.log import setup_logging
import logging
from nixietune import load_dataset_split

setup_logging()
logger = logging.getLogger()


def test_dataset_loading():
    dataset = load_dataset("nixiesearch/ms-marco-dummy", split="train[:10%]")
    expected = Features(
        {
            "query": Value("string"),
            "positive": Sequence(Value("string")),
            "negative": Sequence(Value("string")),
        }
    )
    assert dataset.features == expected


def test_dataset_load_split_file():
    ds = load_dataset_split("tests/data/dummy.jsonl.gz", split="train")
    assert len(ds) == 10


def test_dataset_load_split_dir():
    ds = load_dataset_split("tests/data/", split="train")
    assert len(ds) == 10


def test_dataset_load_split_hf():
    ds = load_dataset_split("nixiesearch/ms-marco-dummy", split="train[:1%]")
    assert len(ds) == 10
