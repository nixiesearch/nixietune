from datasets import load_dataset, Features, Value, Sequence
from nixietune.log import setup_logging
import logging

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
