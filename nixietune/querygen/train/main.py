import os
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from nixietune.querygen.train.trainer import QueryGenTrainer
from nixietune.querygen.train.arguments import QueryGenArguments
from transformers import HfArgumentParser, TrainerCallback
import logging
from nixietune import ModelArguments, DatasetArguments


class EvaluateFirstStepCallback(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_evaluate = True


def main(argv):
    logger = logging.getLogger()
    parser = HfArgumentParser((ModelArguments, DatasetArguments, QueryGenArguments))
    if len(argv) == 2 and argv[1].endswith(".json"):
        model_args, dataset_args, training_args = parser.parse_json_file(json_file=os.path.abspath(argv[1]))
    else:
        model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f"Training parameters: {training_args}")

    trainer = QueryGenTrainer(model_id=model_args.model_name_or_path, args=training_args, dataset_args=dataset_args)
    trainer.train()
    trainer.save_model(training_args.output_dir)
