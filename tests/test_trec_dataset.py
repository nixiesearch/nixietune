from nixietune.format.trec import TRECDatasetReader, TRECDatasetWriter
from transformers import LlamaTokenizer
from nixietune.log import setup_logging
import logging
from datasets import Dataset
import tempfile

setup_logging()
logger = logging.getLogger()


def test_writing():
    source = Dataset.from_dict({"query": ["foo", "bar"], "text": ["baz", "qux"]})
    with tempfile.TemporaryDirectory() as tmpdir:
        TRECDatasetWriter.save(source, tmpdir)


def test_join_qdl():
    ds = TRECDatasetReader("tests/data/trec")
    joined = ds.join_query_doc_score(ds.corpus(), ds.queries(), ds.qrels("qrels/train.tsv")).to_dict()
    assert len(joined["query"]) == 4


def test_corpus_load():
    ds = TRECDatasetReader("tests/data/trec")
    corpus = ds.corpus().to_dict()
    assert len(corpus["_id"]) == 4


def test_query_load():
    ds = TRECDatasetReader("tests/data/trec")
    q = ds.queries().to_dict()
    assert len(q["_id"]) == 2


def test_corpus_load_tokenized():
    tokenizer = LlamaTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    ds = TRECDatasetReader("tests/data/trec", tokenizer=tokenizer)
    corpus = ds.corpus().to_dict()
    assert len(corpus["_id"]) == 4


def test_qrel_load():
    ds = TRECDatasetReader("tests/data/trec")
    q = ds.qrels("qrels/train.tsv").to_dict()
    assert q["query-id"] == ["PLAIN-1", "PLAIN-1", "PLAIN-2", "PLAIN-2"]
