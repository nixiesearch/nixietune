from nixietune.log import setup_logging
import logging
from transformers import HfArgumentParser, AutoModelForCausalLM, LlamaTokenizer
from peft import PeftModel
from dataclasses import dataclass, field
import torch

setup_logging()


@dataclass
class MergeArguments:
    adapter_path: str = field(metadata={"help": "Path to the adapter model"})
    base_model: str = field(metadata={"help": "Name/path of the base model"})
    out_path: str = field(metadata={"help": "path to write the merged model to"})


if __name__ == "__main__":
    logger = logging.getLogger()
    parser = HfArgumentParser((MergeArguments))
    (merge_args,) = parser.parse_args_into_dataclasses()
    tokenizer = LlamaTokenizer.from_pretrained(
        merge_args.base_model,
        add_eos_token=False,
        add_bos_token=False,
        use_fast=False,
        pad_token="<unk>",
        padding_side="right",
    )
    tokenizer.save_pretrained(merge_args.out_path)
    model = AutoModelForCausalLM.from_pretrained(merge_args.base_model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, merge_args.adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(merge_args.out_path)
