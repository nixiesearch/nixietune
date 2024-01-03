from datasets import load_dataset
from nixietune.format import TripletFormat, QueryDocLabelFormat, QueryPosNegsFormat
from transformers import AutoTokenizer
import pytest


@pytest.fixture
def dataset():
    return load_dataset("nixiesearch/ms-marco-dummy", split="train[:10%]")


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def test_triplet_format(dataset, tokenizer):
    parser = TripletFormat(tokenizer)
    _test_format(dataset, parser)


def test_querydoclabel_format(dataset, tokenizer):
    parser = QueryDocLabelFormat(tokenizer)
    _test_format(dataset, parser)


def test_queryposnegs_format(dataset, tokenizer):
    parser = QueryPosNegsFormat(tokenizer)
    _test_format(dataset, parser)


def _test_format(dataset, format):
    schema = format.schema("uint16")
    processed = dataset.map(
        function=format.tokenize,
        batched=True,
        remove_columns=["query", "positive", "negative"],
        features=schema,
    )
    assert processed.features == schema
