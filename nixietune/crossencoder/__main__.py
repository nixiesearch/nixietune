import sys
import os
from nixietune.log import setup_logging
import logging
from transformers import (
    HfArgumentParser,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)
from nixietune.arguments import ModelArguments, DatasetArguments
from nixietune.crossencoder.arguments import CrossEncoderArguments
from nixietune.format.json import JSONDataset
from nixietune.util.eval_callback import EvaluateFirstStepCallback
from nixietune.crossencoder.trainer import CrossEncoderTrainer
import torch
from peft import get_peft_model, LoraConfig
from typing import Tuple
from nixietune.crossencoder.dataset import QueryDocListFormat, QueryDocTupleFormat

setup_logging()
logger = logging.getLogger()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetArguments, CrossEncoderArguments))
    model_config: Tuple[ModelArguments, DatasetArguments, CrossEncoderArguments]
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_config = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_config = parser.parse_args_into_dataclasses()
    model_args, dataset_args, training_args = model_config
    device = "cpu" if training_args.use_cpu else "cuda"
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, add_eos_token=True, add_bos_token=True)
    match training_args.layout:
        case "cross":
            model_config.num_labels = 1
            format = QueryDocTupleFormat(tokenizer, training_args.seq_len)
        case "rank":
            model_config.num_labels = training_args.num_negatives + 1
            format = QueryDocListFormat(tokenizer, training_args.seq_len)
        case other:
            raise Exception(f"layout {training_args.layout} not supported")
    if training_args.lora:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True if training_args.lora_load_bits == 8 else False,
            load_in_4bit=True if training_args.lora_load_bits == 4 else False,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation=training_args.attn_implementation,
        )
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head"],
            lora_dropout=training_args.lora_dropout,
        )
        model = get_peft_model(model, lora_config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            attn_implementation=training_args.attn_implementation,
        )

    train = JSONDataset.load(
        dataset_args.train_dataset,
        split=dataset_args.train_split,
        num_workers=training_args.dataloader_num_workers,
    )
    test = None
    if (dataset_args.eval_dataset) is not None:
        test = JSONDataset.load(
            dataset_args.eval_dataset,
            split=dataset_args.eval_split,
            num_workers=training_args.dataloader_num_workers,
        )
    logger.info(f"Training parameters: {training_args}")

    trainer = CrossEncoderTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        eval_metrics=training_args.eval_metrics,
        fmt=format,
    )
    # if test is not None:
    #     if trainer.is_deepspeed_enabled:
    #         logger.info("Not running eval before train: deepspeed is enabled (and not yet fully initialized)")
    #     else:
    #         logger.info(trainer.evaluate())
    #     trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train()
    model.save_pretrained(save_directory=training_args.output_dir)
    tokenizer.save_pretrained(save_directory=training_args.output_dir)
