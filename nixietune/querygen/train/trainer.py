from typing import Optional
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
from nixietune.querygen.train.arguments import QueryGenArguments
from datasets import Dataset
import torch
import logging
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os
from nixietune.querygen.train.tokenizer import DatasetTokenizer

logger = logging.getLogger()


class QueryGenTrainer(SFTTrainer):
    def __init__(
        self,
        model_id: str,
        args: QueryGenArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        **kwargs,
    ) -> None:
        self.model_id = model_id
        self.args = args
        self.args.remove_unused_columns = False
        self.args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        tokenizer = LlamaTokenizer.from_pretrained(
            model_id, add_eos_token=False, add_bos_token=False, use_fast=False, pad_token="<unk>", padding_side="left"
        )
        tokenizer.pad_token = "<unk>"
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        # self.args.evaluation_strategy = "no" if eval_dataset is None

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

        dt = DatasetTokenizer(tokenizer, args.seq_len)

        super().__init__(
            args=self.args,
            model=self.model,
            max_seq_length=self.args.seq_len,
            train_dataset=train_dataset.map(
                function=dt.tokenize, batched=True, remove_columns=["query", "passage"], num_proc=14
            ),
            eval_dataset=eval_dataset.map(
                function=dt.tokenize, batched=True, remove_columns=["query", "passage"], num_proc=14
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
