import os
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from nixietune.querygen.train.trainer import QueryGenTrainer
from nixietune.querygen.train.arguments import QueryGenArguments
from transformers import HfArgumentParser, TrainerCallback
import logging
from nixietune import load_dataset_split, ModelArguments, DatasetArguments


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

    train = load_dataset_split(
        dataset_args.train_dataset,
        split=dataset_args.train_split,
        samples=dataset_args.train_samples,
        streaming=dataset_args.streaming,
        schema=None,
    )
    if dataset_args.eval_dataset is not None:
        test = load_dataset_split(
            dataset_args.eval_dataset,
            split=dataset_args.eval_split,
            samples=dataset_args.eval_samples,
            streaming=False,
            schema=None,
        )
    else:
        test = None

    logger.info(f"Training parameters: {training_args}")

    trainer = QueryGenTrainer(
        model_id=model_args.model_name_or_path, args=training_args, train_dataset=train, eval_dataset=test
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
