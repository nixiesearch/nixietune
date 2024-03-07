from nixietune.querygen.train.trainer import QueryGenTrainer
from transformers import LlamaTokenizer


def test_data_load():
    tok = LlamaTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        add_eos_token=False,
        add_bos_token=False,
        use_fast=False,
        pad_token="<unk>",
        padding_side="left",
    )
    ds = QueryGenTrainer.load_dataset(
        name="tests/querygen/train/data/corpus.jsonl",
        split="train",
        tokenizer=tok,
        seq_len=128,
        num_workers=1,
    ).to_dict()
    assert ds == {
        "input_ids": [[1, 264, 5709, 28747, 287, 2]],
        "attention_mask": [[1, 1, 1, 1, 1, 1]],
    }
