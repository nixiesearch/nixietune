import os
from sentence_transformers import SentenceTransformer
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from nixietune.biencoder import BiencoderTrainer, BiencoderTrainingArguments
from transformers import HfArgumentParser, TrainerCallback
import logging
from nixietune import ModelArguments, DatasetArguments
from nixietune.format.jsontokenized import JSONTokenizedDataset


class EvaluateFirstStepCallback(TrainerCallback):
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
    train = JSONTokenizedDataset.load(
        dataset_args.train_dataset,
        split=dataset_args.train_split,
        max_len=training_args.seq_len,
        tok=model.tokenizer,
        num_workers=training_args.dataloader_num_workers,
    )
    test = None
    if (dataset_args.eval_dataset) is not None:
        test = JSONTokenizedDataset.load(
            dataset_args.eval_dataset,
            split=dataset_args.eval_split,
            max_len=training_args.seq_len,
            tok=model.tokenizer,
            num_workers=training_args.dataloader_num_workers,
        )
    logger.info(f"Training parameters: {training_args}")

    trainer = BiencoderTrainer(
        model=model,
        tokenizer=model.tokenizer,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        train_split=dataset_args.train_split,
        eval_split=dataset_args.eval_split,
        eval_metrics=training_args.eval_metrics,
    )
    if test is not None:
        if trainer.is_deepspeed_enabled:
            logger.info("Not running eval before train: deepspeed is enabled (and not yet fully initialized)")
        else:
            logger.info(trainer.evaluate())
        trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train()

    model.save(path=training_args.output_dir)
