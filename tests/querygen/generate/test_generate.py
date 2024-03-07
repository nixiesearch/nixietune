from nixietune.querygen.generate.generate import QueryGenerator
from transformers import LlamaTokenizer


def test_load():
    tok = LlamaTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        add_eos_token=False,
        add_bos_token=False,
        use_fast=False,
        pad_token="<unk>",
        padding_side="left",
    )
    result = QueryGenerator.load_dataset(
        input="tests/querygen/generate/data/corpus.jsonl",
        tokenizer=tok,
        seq_len=128,
        num_workers=1,
        batch_size=1,
        device="cpu",
    )
    print(result)
