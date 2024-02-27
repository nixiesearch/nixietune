import os
from sentence_transformers import SentenceTransformer
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from nixietune.biencoder import BiencoderTrainer, BiencoderTrainingArguments
from transformers import HfArgumentParser, TrainerCallback, AutoTokenizer
import logging
from nixietune import ModelArguments, DatasetArguments
from nixietune.format.trec import TRECDataset
from datasets import Features, Value


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
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = SentenceTransformer(model_args.model_name_or_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    if dataset_args.train_dataset == dataset_args.eval_dataset:
        train = TRECDataset.from_dir(
            path=dataset_args.train_dataset,
            tokenizer=tokenizer,
            qrel_splits=[dataset_args.train_split, dataset_args.eval_split],
            max_length=training_args.seq_len,
        )
        test = train
    else:
        train = TRECDataset.from_dir(
            path=dataset_args.train_dataset,
            tokenizer=tokenizer,
            qrel_splits=[dataset_args.train_split],
            max_length=training_args.seq_len,
        )
        if (dataset_args.eval_dataset) is not None:
            test = TRECDataset.from_dir(
                path=dataset_args.train_dataset,
                tokenizer=tokenizer,
                qrel_splits=[dataset_args.test_split],
                max_length=training_args.seq_len,
            )
        else:
            test = None

    logger.info(f"Training parameters: {training_args}")

    trainer = BiencoderTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        train_split=dataset_args.train_split,
        eval_split=dataset_args.eval_split,
    )
    if test is not None:
        if trainer.is_deepspeed_enabled:
            logger.info("Not running eval before train: deepspeed is enabled (and not yet fully initialized)")
        else:
            logger.info(trainer.evaluate())
        trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train()

    model.save(path=training_args.output_dir)
