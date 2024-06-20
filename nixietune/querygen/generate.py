from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    pipeline,
)
import torch
from typing import Dict, List, Any
from nixietune.format.json import JSONDataset
from tqdm import tqdm
import itertools
import json


@dataclass
class GeneratorArguments:
    model_name_or_path: str = field(metadata={"help": "model path"})
    seq_len: int = field(default=512, metadata={"help": "sequence length of the input passage"})
    max_new_tokens: int = field(default=32, metadata={"help": "how many new tokens should be generated max"})
    batch_size: int = field(default=32, metadata={"help": "batch size"})
    num_workers: int = field(default=8, metadata={"help": "number of data loader workers"})
    strategy: str = field(default="greedy", metadata={"help": "sampling strategy: greedy/beam=N"})


@dataclass
class DatasetArguments:
    input_file: str = field(metadata={"help": "path to input dataset"})
    output_file: str = field(metadata={"help": "path to output file"})


class QueryGenerator:
    def __init__(self, args: GeneratorArguments) -> None:
        self.args = args
        self.generator = pipeline(
            task="text-generation",
            model=args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def generate(self, input: str, output: str):
        corpus = JSONDataset.from_file(input, num_workers=self.args.num_workers, split="train").select_columns(["doc"])
        prompter = Prompter(self.args.model_name_or_path, self.args.seq_len)
        prompts = corpus.map(
            function=prompter.make_prompt,
            batched=True,
            desc="formatting prompts",
            num_proc=self.args.num_workers,
        )
        dataset_dict = prompts.to_dict()
        dataset = []
        for doc, prompt, length in zip(dataset_dict["doc"], dataset_dict["prompt"], dataset_dict["length"]):
            dataset.append({"doc": doc, "prompt": prompt, "length": length})
        ds2 = sorted(dataset, key=lambda x: x["length"], reverse=True)
        match self.args.strategy:
            case "greedy":
                gen_kwargs = {}
            case other if other.startswith("beam="):
                gen_kwargs = {"do_sample": True, "num_beams": int(other.split("=")[1])}
        print(gen_kwargs)
        with open(output, "w") as file:
            for batch in tqdm(list(itertools.batched(ds2, self.args.batch_size)), unit_scale=self.args.batch_size):
                out = self.generator(
                    text_inputs=[item["prompt"] for item in batch],
                    batch_size=self.args.batch_size,
                    return_full_text=True,
                    max_new_tokens=self.args.max_new_tokens,
                    num_return_sequences=1,
                    **gen_kwargs,
                )
                for item in out:
                    raw = item[0]["generated_text"]
                    doc = prompter.parse_doc(raw)
                    query = prompter.parse_response(raw)
                    file.write(json.dumps({"doc": doc, "query": query}) + "\n")


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
