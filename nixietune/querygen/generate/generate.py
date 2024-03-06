from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, GenerationConfig, PreTrainedTokenizerBase
import torch
from typing import Dict, Generator, List, Any
from nixietune.format.json import JSONDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class GeneratorArguments:
    model_name_or_path: str = field(metadata={"help": "model path"})
    seq_len: int = field(default=512, metadata={"help": "sequence length of the input passage"})
    prompt_modifier: str = field(default="", metadata={"help": "prompt prefix modifiers"})
    max_new_tokens: int = field(default=32, metadata={"help": "how many new tokens should be generated max"})
    batch_size: int = field(default=48, metadata={"help": "batch size"})
    num_workers: int = field(default=8, metadata={"help": "number of data loader workers"})


@dataclass
class DatasetArguments:
    input_file: str = field(metadata={"help": "path to input dataset"})
    output_file: str = field(metadata={"help": "path to output file"})


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

    def generate(self, input: str):
        pt = PromptTokenizer(self.tokenizer, self.args.seq_len, self.model.device)
        corpus = JSONDataset.from_file(
            input,
            tokenizer=self.tokenizer,
            max_len=256,
            split="train",
            num_workers=self.args.num_workers,
        ).select_columns(["pos"])
        processed = corpus.map(
            function=pt.tokenize_batch,
            batched=True,
            desc="formatting prompts",
            remove_columns=["pos"],
            num_proc=self.args.num_workers,
        )
        loader = DataLoader(
            processed,
            batch_size=self.args.batch_size,
            collate_fn=pt.collate_batch,
        )

        for batch in tqdm(loader, desc="generating queries"):
            for item in self.process_batch(batch):
                yield item

    def process_batch(self, batch) -> List[Dict[str, Any]]:
        config = GenerationConfig(
            max_new_tokens=self.args.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        outputs = self.model.generate(**batch, generation_config=config)
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        passages = []
        queries = []
        for out in decoded:
            pos = out.index("query: ")
            queries.append(out[pos + 7 :])
            passages.append(out[:pos])

        return [{"query": query, "pos": doc} for doc, query in zip(passages, queries)]


class PromptTokenizer:
    def __init__(self, tok: PreTrainedTokenizerBase, seq_len: int, device):
        self.tokenizer = tok
        self.seq_len = seq_len
        self.query_input_ids = self.tokenizer(" query:", padding=False)["input_ids"]  # no space at end!
        self.device = device

    def collate_batch(self, batch: List[Dict[str, Any]]):
        passages_inputs = [item["input_ids"] for item in batch]
        passages_attmasks = [item["attention_mask"] for item in batch]
        encoded = BatchEncoding({"input_ids": passages_inputs, "attention_mask": passages_attmasks})
        padded = self.tokenizer.pad(encoded, pad_to_multiple_of=8, return_tensors="pt").to(self.device)
        return padded

    def tokenize_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, List]:
        tokenized_passages = batch["pos"]
        max_doc_len = self.seq_len - len(self.query_input_ids) - 1
        passages_inputs = []
        passages_attmasks = []
        for tp in tokenized_passages:
            passage = (
                [self.tokenizer.bos_token_id] + tp[:max_doc_len] + self.query_input_ids
            )  # no eos, as we expect to continue the generation
            passages_inputs.append(passage)
            passages_attmasks.append([1] * len(passage))
        return {"input_ids": passages_inputs, "attention_mask": passages_attmasks}
