from dataclasses import dataclass, field
import os
import sys
from sentence_transformers import SentenceTransformer
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from nixietune.trainer import BiencoderTrainer, BiencoderTrainingArguments
from nixietune.log import setup_logging
from datasets import load_dataset, concatenate_datasets
from transformers import HfArgumentParser, TrainerCallback
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


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_evaluate = True


def main():
    logger = logging.getLogger()
    parser = HfArgumentParser((ModelArguments, DatasetArguments, BiencoderTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, dataset_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    model = SentenceTransformer(model_args.model_name_or_path)
    # train = load_dataset(dataset_args.train_dataset, split="train")
    # test = load_dataset(dataset_args.eval_dataset, split="test")
    train = load_dataset(
        "json", data_files={"train": "/home/shutty/data/mneg/data/train/train.jsonl.zst"}, split="train"
    )
    test = load_dataset("json", data_files={"test": "/home/shutty/data/mneg/data/test/test.jsonl.zst"}, split="test")

    logger.info(f"Training parameters: {training_args}")

    trainer = BiencoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
    )
    trainer.add_callback(EvaluateFirstStepCallback())
    trainer.train()

    model.save(path="./model-pretrain-infonce8-cos-warm")


if __name__ == "__main__":
    main()
