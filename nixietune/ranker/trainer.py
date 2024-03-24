from typing import Optional
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer, PreTrainedTokenizerBase
from nixietune.ranker.arguments import RankerArguments
from datasets import Dataset
import torch
import logging
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from nixietune.arguments import DatasetArguments
from nixietune.format.json import JSONDataset
from dataclasses import dataclass
from typing import Dict, List
import random
from nixietune.metrics.callback import EvalMetrics

logger = logging.getLogger()


class RankerTrainer(SFTTrainer):
    def __init__(
        self,
        model_id: str,
        args: RankerArguments,
        dataset_args: DatasetArguments,
        eval_metrics: List[str],
        **kwargs,
    ) -> None:
        self.model_id = model_id
        self.args = args
        self.args.remove_unused_columns = False
        self.args.gradient_checkpointing_kwargs = {"use_reentrant": False}

        tokenizer = LlamaTokenizer.from_pretrained(model_id, use_fast=False, pad_token="<unk>", padding_side="left")
        tokenizer.pad_token = "<unk>"
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.eval_metrics = EvalMetrics(eval_metrics, tokenizer)

        bnb_config = BitsAndBytesConfig(
            # load_in_8bit=True
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head"],
            lora_dropout=0.05,
        )

        super().__init__(
            args=self.args,
            model=self.model,
            max_seq_length=self.args.seq_len,
            train_dataset=RankerTrainer.load_dataset(
                name=dataset_args.train_dataset,
                split=dataset_args.train_split,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                doc_seq_len=args.doc_seq_len,
                num_workers=args.dataloader_num_workers,
                num_negatives=args.num_negatives,
            ),
            eval_dataset=RankerTrainer.load_dataset(
                name=dataset_args.eval_dataset,
                split=dataset_args.eval_split,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                doc_seq_len=args.doc_seq_len,
                num_workers=args.dataloader_num_workers,
                num_negatives=args.num_negatives,
            ),
            peft_config=lora_config,
            tokenizer=self.tokenizer,
            packing=False,
            dataset_text_field="text",
            compute_metrics=self.eval_metrics.compute_ranker,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
            data_collator=DataCollatorForCompletionOnlyLM(response_template="<label>", tokenizer=self.tokenizer),
            **kwargs,
        )

    def preprocess_logits_for_metrics(self, logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def _prepare_dataset(
        self,
        dataset,
        tokenizer,
        packing,
        dataset_text_field,
        max_seq_length,
        formatting_func,
        num_of_sequences,
        chars_per_token,
        remove_unused_columns=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        return dataset

    @staticmethod
    def load_dataset(
        name: Optional[str],
        split: Optional[str],
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        doc_seq_len: int,
        num_negatives: int,
        num_workers: int,
    ) -> Optional[Dataset]:
        if name:
            split_str = split if split else "train"
            loaded = JSONDataset.load(path=name, split=split_str, num_workers=num_workers)
            fmt = RankerDataset(tokenizer, seq_len, doc_seq_len, num_negatives)
            processed = loaded.map(
                function=fmt.tokenize,
                batched=True,
                remove_columns=["query", "doc", "neg", "negscore"],
                num_proc=num_workers,
                desc=f"Formatting {split_str} prompts",
            )
            return processed
        else:
            return None


@dataclass
class RankerDataset:
    tokenizer: PreTrainedTokenizerBase
    seq_len: int
    doc_seq_len: int
    num_negatives: int

    def tokenize(self, batch: Dict[str, List]) -> Dict[str, List]:
        all_negs = [neg for negs in batch["neg"] for neg in negs]
        strings = batch["query"] + batch["doc"] + all_negs
        tokenized = self.tokenizer(strings, padding=False, truncation=True, max_length=self.doc_seq_len)
        trimmed = {
            str: self.tokenizer.decode(tok, skip_special_tokens=True)
            for str, tok in zip(strings, tokenized["input_ids"])
        }
        prompts = []
        for query, pos, negs in zip(batch["query"], batch["doc"], batch["neg"]):
            neg_sample = random.choices([trimmed[neg] for neg in negs], k=min(len(negs), self.num_negatives))
            docs = [trimmed[pos]] + neg_sample
            random.shuffle(docs)
            pos_index = docs.index(trimmed[pos])
            prompt = ["<query>", trimmed[query]]
            for i, doc in enumerate(docs):
                prompt.append("<" + self.index(i) + ">")
                prompt.append(doc)
            prompt.append("<label>" + self.index(pos_index))  # no space!
            prompts.append(" ".join(prompt))
        tp = self.tokenizer(prompts, padding=False, truncation=True, max_length=self.seq_len)
        # return {"text": prompts}
        return {"input_ids": tp["input_ids"], "attention_mask": tp["attention_mask"]}

    def index(self, num: int) -> str:
        if num >= 0 and num <= 9:
            return str(num)
        elif num >= 10:
            return chr(87 + num)
