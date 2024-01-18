import os
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from nixietune.querygen.train.trainer import QueryGenTrainer
from nixietune.querygen.train.arguments import QueryGenArguments
from transformers import HfArgumentParser, TrainerCallback
import logging
from nixietune import load_dataset_split, ModelArguments, DatasetArguments
from nixietune.format.trec import TRECDatasetReader


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

    trt = TRECDatasetReader(dataset_args.train_dataset)
    train = trt.join_query_doc_score(corpus=trt.corpus(), queries=trt.queries(), qrels=trt.qrels("qrels.tsv"))
    train = train.select_columns(["query", "passage"])
    if dataset_args.eval_dataset is not None:
        tet = TRECDatasetReader(dataset_args.eval_dataset)
        test = trt.join_query_doc_score(corpus=tet.corpus(), queries=tet.queries(), qrels=tet.qrels("qrels.tsv"))
        test = test.select_columns(["query", "passage"])
    else:
        test = None

    logger.info(f"Training parameters: {training_args}")

    trainer = QueryGenTrainer(
        model_id=model_args.model_name_or_path, args=training_args, train_dataset=train, eval_dataset=test
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
