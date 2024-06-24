from transformers import AutoTokenizer
from datasets import Dataset
from typing import Dict, Any, List


class Prompter:
    def __init__(self, model, seq_len):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.seq_len = seq_len

    def make_prompt(self, batch: Dict[str, List[Any]]) -> Dict[str, List]:
        trimmed = self.tokenizer(batch["doc"], padding=False, truncation=True, max_length=self.seq_len)
        decoded = self.tokenizer.batch_decode(trimmed["input_ids"], skip_special_tokens=True)
        prompt = "### Instruction:\nWrite a short query which can be used to search a given document:\n\n### Input:\n{input}\n\n### Response:\n"
        return {
            "prompt": [prompt.format(input=doc) for doc in decoded],
            "length": [len(doc) for doc in trimmed["input_ids"]],
        }

    def parse_doc(self, input: str) -> str:
        left = "### Input:\n"
        right = "\n\n### Response:\n"
        start_index = input.find(left)
        end_index = input.find(right)
        return input[start_index + len(left) : end_index]

    def parse_response(self, input: str) -> str:
        right = "\n\n### Response:\n"
        start_index = input.find(right)
        return input[start_index + len(right) :].strip()


class QueryGenDataset:
    @staticmethod
    def from_dataset(corpus: Dataset, prompter: Prompter, num_workers: int) -> Dataset:
        processed_prompts = corpus.map(
            function=prompter.make_prompt,
            batched=True,
            desc="formatting prompts",
            num_proc=num_workers,
        )
        sorted_prompts = processed_prompts.sort(column_names="length", reverse=True)
        return sorted_prompts
