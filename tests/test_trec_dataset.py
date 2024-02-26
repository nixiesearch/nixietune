from nixietune.format.trec import TRECDataset
from transformers import AutoTokenizer
from nixietune.log import setup_logging
import logging
from datasets import Dataset
from typing import Dict, List

setup_logging()
logger = logging.getLogger()


def test_pairs():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = TRECDataset.from_dir(path="tests/data/trec", tokenizer=tokenizer)
    joined = ds.as_tokenized_pairs().to_dict()
    assert len(joined["query"]) == 4


def test_triplets():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = TRECDataset.from_dir(path="tests/data/trec", tokenizer=tokenizer)
    joined = ds.as_tokenized_triplets()
    logger.info(joined.features)
    assert len(joined.to_dict()["query"]) == 4


# def test_join_qpn():
#     ds = TRECDatasetReader("tests/data/trec")
#     joined = ds.join_query_pos_neg(ds.corpus(), ds.queries(), ds.qrels("qrels/train.tsv")).to_dict()
#     assert len(joined["query"]) == 4


# def test_corpus_load():
#     ds = TRECDatasetReader("tests/data/trec")
#     corpus = ds.corpus().to_dict()
#     assert len(corpus["_id"]) == 4


# def test_query_load():
#     ds = TRECDatasetReader("tests/data/trec")
#     q = ds.queries().to_dict()
#     assert len(q["_id"]) == 2


# def test_corpus_load_tokenized():
#     tokenizer = LlamaTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
#     ds = TRECDatasetReader("tests/data/trec", tokenizer=tokenizer)
#     corpus = ds.corpus().to_dict()
#     assert len(corpus["_id"]) == 4


# def test_qrel_load():
#     ds = TRECDatasetReader("tests/data/trec")
#     q = ds.qrels("qrels/train.tsv").to_dict()
#     assert q["query-id"] == ["PLAIN-1", "PLAIN-1", "PLAIN-2", "PLAIN-2"]
