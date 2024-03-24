from nixietune.ranker.trainer import RankerDataset
from transformers import LlamaTokenizer
from datasets import Dataset
from nixietune.format.json import JSONDataset
import random

random.seed(42)


def test_dataset():
    tokenizer = LlamaTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        add_eos_token=False,
        add_bos_token=False,
        use_fast=False,
        pad_token="<unk>",
        padding_side="left",
    )
    fmt = RankerDataset(tokenizer, seq_len=1024, doc_seq_len=128, num_negatives=2)
    data = Dataset.from_list([{"query": "hello", "doc": "cruel world", "neg": ["no"]}])
    parsed = JSONDataset.from_dataset(ds=data)
    processed = parsed.map(function=fmt.tokenize, batched=True).to_dict()
    decoded = tokenizer.decode(processed["input_ids"][0], skip_special_tokens=False)
    assert decoded == "<query> hello <0> no <1> cruel world <label>1"
