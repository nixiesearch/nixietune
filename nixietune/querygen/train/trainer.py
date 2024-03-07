from typing import Optional
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer, PreTrainedTokenizerBase
from nixietune.querygen.train.arguments import QueryGenArguments
from datasets import Dataset
import torch
import logging
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os
from nixietune.querygen.train.tokenizer import DatasetTokenizer
from nixietune.arguments import DatasetArguments
from nixietune.format.json import JSONDataset

logger = logging.getLogger()


class QueryGenTrainer(SFTTrainer):
    def __init__(
        self,
        model_id: str,
        args: QueryGenArguments,
        dataset_args: DatasetArguments,
        **kwargs,
    ) -> None:
        self.model_id = model_id
        self.args = args
        self.args.remove_unused_columns = False
        self.args.gradient_checkpointing_kwargs = {"use_reentrant": False}

        tokenizer = LlamaTokenizer.from_pretrained(
            model_id, add_eos_token=False, add_bos_token=False, use_fast=False, pad_token="<unk>", padding_side="left"
        )
        tokenizer.pad_token = "<unk>"
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

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
            train_dataset=QueryGenTrainer.load_dataset(
                name=dataset_args.train_dataset,
                split=dataset_args.train_split,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                num_workers=args.dataloader_num_workers,
            ),
            eval_dataset=QueryGenTrainer.load_dataset(
                name=dataset_args.train_dataset,
                split=dataset_args.train_split,
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                num_workers=args.dataloader_num_workers,
            ),
            peft_config=lora_config,
            tokenizer=self.tokenizer,
            packing=False,
            dataset_text_field="text",
            data_collator=DataCollatorForCompletionOnlyLM(response_template="query:", tokenizer=self.tokenizer),
        )

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
        num_workers: int,
    ) -> Optional[Dataset]:
        if name:
            dt = DatasetTokenizer(tokenizer, seq_len)
            split_str = split if split else "train"
            loaded = JSONDataset.load(path=name, split=split_str, num_workers=num_workers)
            processed = loaded.map(
                function=dt.tokenize,
                batched=True,
                remove_columns=["query", "doc", "neg", "negscore"],
                num_proc=num_workers,
                desc=f"Formatting {split_str} prompts",
            )
            return processed
        else:
            return None
