from dataclasses import dataclass, field
from transformers import (
    pipeline,
)
import torch
from nixietune.format.json import JSONDataset
import json
from transformers.pipelines.pt_utils import KeyDataset
from nixietune.querygen.dataset import Prompter, QueryGenDataset
from tqdm import tqdm


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
        prompts = QueryGenDataset.from_dataset(corpus=corpus, prompter=prompter, num_workers=self.args.num_workers)

        match self.args.strategy:
            case "greedy":
                gen_kwargs = {}
            case other if other.startswith("beam="):
                gen_kwargs = {"do_sample": True, "num_beams": int(other.split("=")[1])}

        with open(output, "w") as file:
            for out in tqdm(
                self.generator(
                    KeyDataset(prompts, "prompt"),
                    batch_size=self.args.batch_size,
                    return_full_text=True,
                    max_new_tokens=self.args.max_new_tokens,
                    num_return_sequences=1,
                    **gen_kwargs,
                ),
                total=len(prompts),
                desc="Generating queries",
            ):
                for item in out:
                    raw = item["generated_text"]
                    doc = prompter.parse_doc(raw)
                    query = prompter.parse_response(raw)
                    file.write(json.dumps({"doc": doc, "query": query}) + "\n")
