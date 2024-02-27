from nixietune.format.trec import TRECDataset
from transformers import AutoTokenizer
from nixietune.log import setup_logging
import logging
from datasets import Dataset
from typing import Dict, List

setup_logging()
logger = logging.getLogger()


def test_corpus():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = TRECDataset.corpus_from_dir(path="tests/data/trec", tokenizer=tokenizer)
    assert len(ds.to_dict()["_id"]) == 4


def test_triplets():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = TRECDataset.from_dir(path="tests/data/trec", tokenizer=tokenizer)
    joined = ds.load_split(split="train")
    logger.info(joined.features)
    assert len(joined.to_dict()["query"]) == 2


# def test_triplets2():
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     ds = TRECDataset.from_dir(path="/home/shutty/data/beir/trec-covid", tokenizer=tokenizer, qrel_splits=["test"])
#     joined = ds.load_split(split="test")
#     assert len(joined.to_dict()["query-id"]) == 50
