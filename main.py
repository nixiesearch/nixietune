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
class DatasetArguments:
    train_dataset: str = field(metadata={"help": "Path to training dataset"})
    eval_dataset: str = field(metadata={"help": "Path to evaluation dataset"})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


def main():
    logger = logging.getLogger()
    parser = HfArgumentParser((ModelArguments, DatasetArguments, BiencoderTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, dataset_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    model = SentenceTransformer(model_args.model_name_or_path)
    dataset = load_dataset(
        "json",
        data_files={"train": dataset_args.train_dataset, "test": dataset_args.eval_dataset},
        num_proc=8,
    )

    logger.info(f"Training parameters: {training_args}")

    trainer = BiencoderTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    trainer.train()


if __name__ == "__main__":
    main()
