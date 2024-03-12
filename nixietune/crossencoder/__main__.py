import sys
import os
from nixietune.log import setup_logging
import logging
from transformers import HfArgumentParser, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from nixietune.arguments import ModelArguments, DatasetArguments
from nixietune.crossencoder.arguments import CrossEncoderArguments
from nixietune.format.json import JSONDataset
from nixietune.util.eval_callback import EvaluateFirstStepCallback
from nixietune.crossencoder.trainer import CrossEncoderTrainer

setup_logging()
logger = logging.getLogger()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetArguments, CrossEncoderArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, dataset_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, dataset_args, training_args = parser.parse_args_into_dataclasses()

    device = "cpu" if training_args.use_cpu else "cuda"
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

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
    )
    if test is not None:
        if trainer.is_deepspeed_enabled:
            logger.info("Not running eval before train: deepspeed is enabled (and not yet fully initialized)")
        else:
            logger.info(trainer.evaluate())
        trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train()

    model.save(path=training_args.output_dir)
