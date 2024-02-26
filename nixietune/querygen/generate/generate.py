from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, GenerationConfig
import torch
from datasets import Dataset
from typing import Dict, List
from nixietune.format.trec import TRECDataset


@dataclass
class GeneratorArguments:
    model_name_or_path: str = field(metadata={"help": "model path"})
    seq_len: int = field(default=512, metadata={"help": "sequence length of the input passage"})
    prompt_modifier: str = field(default="", metadata={"help": "prompt prefix modifiers"})
    max_new_tokens: int = field(default=32, metadata={"help": "how many new tokens should be generated max"})


@dataclass
class DatasetArguments:
    input_path: str = field(metadata={"help": "path to input dataset"})
    output_path: str = field(metadata={"help": "path to output dir"})


class QueryGenerator:
    def __init__(self, args: GeneratorArguments) -> None:
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            add_eos_token=False,
            add_bos_token=False,
            use_fast=False,
            pad_token="<unk>",
            padding_side="left",
        )
        self.tokenizer.pad_token = "<unk>"
        model_kwargs = {}
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
        self.model.eval()

    def generate(self, input: str) -> Dataset:
        trec = TRECDataset.from_dir(input)
        dataset = trec.corpus.select_columns(["text", "_id"])
        processed = dataset.map(function=self.process_batch, batched=True, batch_size=24)
        return processed

    def process_batch(self, batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
        query = self.tokenizer(f" {self.args.prompt_modifier} query:", padding=False)  # no space at end!
        query_input_ids = query["input_ids"]
        tokenized_passages = self.tokenizer(
            batch["text"], padding=False, max_length=self.args.seq_len, truncation=True
        )
        max_doc_len = self.args.seq_len - len(query_input_ids) - 1
        passages_inputs = []
        passages_attmasks = []
        for tp in tokenized_passages["input_ids"]:
            passage = (
                [self.tokenizer.bos_token_id] + tp[:max_doc_len] + query_input_ids
            )  # no eos, as we expect to continue the generation
            passages_inputs.append(passage)
            passages_attmasks.append([1] * len(passage))
        encoded = BatchEncoding({"input_ids": passages_inputs, "attention_mask": passages_attmasks})
        padded = self.tokenizer.pad(
            encoded, max_length=self.args.seq_len, pad_to_multiple_of=8, return_tensors="pt"
        ).to(self.model.device)
        config = GenerationConfig(
            max_new_tokens=self.args.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        outputs = self.model.generate(**padded, generation_config=config)
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        queries = []
        for out in decoded:
            pos = out.index("query: ")
            queries.append(out[pos + 7 :])
        return {"text": batch["text"], "query": queries, "_id": batch["_id"]}
