from datasets import Dataset

from nixietune.querygen.dataset import Prompter, QueryGenDataset


def test_dataset():
    source = Dataset.from_dict({"doc": ["hello, world"]})
    prompter = Prompter("mistralai/Mistral-7B-v0.3", 32)
    prompts = QueryGenDataset.from_dataset(source, prompter, None)
    assert prompts.to_dict() == {
        "doc": ["hello, world"],
        "prompt": [
            "### Instruction:\nWrite a short query which can be used to search a given document:\n\n### Input:\nhello, world\n\n### Response:\n"
        ],
        "length": [5],
    }
