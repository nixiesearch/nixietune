from datasets import load_dataset, Features, Value
from nixietune.log import setup_logging
from nixietune.format import QueryDocLabelFormat
from transformers import AutoTokenizer
import logging

setup_logging()
logger = logging.getLogger()


def test_querydoclabel_tokenizer():
    dataset = load_dataset("nixiesearch/ms-marco-dummy", split="train[:10%]")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    parser = QueryDocLabelFormat(tokenizer)
    schema = parser.schema("uint16")
    processed = dataset.map(
        function=parser.tokenize,
        batched=True,
        remove_columns=["query", "positive", "negative"],
        features=schema,
    )
    assert processed.features == schema
