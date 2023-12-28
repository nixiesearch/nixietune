from dataclasses import dataclass, field
import os
import sys
from sentence_transformers import SentenceTransformer
from nixietune.trainer import BiencoderTrainer, BiencoderTrainingArguments
from nixietune.log import setup_logging
from datasets import load_dataset
from transformers import HfArgumentParser
import logging

setup_logging()


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


def main():
    logger = logging.getLogger()
    parser = HfArgumentParser((ModelArguments, BiencoderTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    model = SentenceTransformer(model_args.model_name_or_path)
    train = load_dataset("nixiesearch/ms-marco-hard-negatives", split="train")
    test = load_dataset("nixiesearch/ms_marco", split="test")
    logger.info(f"Training parameters: {training_args}")

    trainer = BiencoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
    )
    trainer.train()


if __name__ == "__main__":
    main()
