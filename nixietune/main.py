import os
from sentence_transformers import SentenceTransformer
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from nixietune.biencoder import BiencoderTrainer, BiencoderTrainingArguments
from transformers import HfArgumentParser, TrainerCallback
import logging
from nixietune import load_dataset_split, ModelArguments, DatasetArguments


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_evaluate = True


def main(argv):
    logger = logging.getLogger()
    parser = HfArgumentParser((ModelArguments, DatasetArguments, BiencoderTrainingArguments))
    if len(argv) == 2 and argv[1].endswith(".json"):
        model_args, dataset_args, training_args = parser.parse_json_file(json_file=os.path.abspath(argv[1]))
    else:
        model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    device = "cpu" if training_args.use_cpu else "cuda"
    model = SentenceTransformer(model_args.model_name_or_path, device=device)
    train = load_dataset_split(
        dataset_args.train_dataset,
        split=dataset_args.train_split,
        samples=dataset_args.train_samples,
    )
    if dataset_args.eval_dataset is not None:
        test = load_dataset_split(
            dataset_args.eval_dataset,
            split=dataset_args.eval_split,
            samples=dataset_args.eval_samples,
        )
    else:
        test = None

    logger.info(f"Training parameters: {training_args}")

    trainer = BiencoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
    )
    if test is not None:
        trainer.add_callback(EvaluateFirstStepCallback())
    trainer.train()

    model.save(path=training_args.output_dir)
