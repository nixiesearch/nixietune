from transformers import Trainer, AutoModelForSequenceClassification, PreTrainedTokenizerBase
from nixietune.crossencoder.arguments import CrossEncoderArguments
from datasets import Dataset
from typing import List, Optional, Dict, Any, Union, Tuple
from nixietune.metrics.callback import EvalMetrics
import torch
from torch import nn
import logging
from nixietune.crossencoder.dataset import CrossEncoderFormat

logger = logging.getLogger()


class CrossEncoderTrainer(Trainer):
    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: PreTrainedTokenizerBase,
        args: CrossEncoderArguments,
        fmt: CrossEncoderFormat,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        eval_metrics: List[str] = ["ndcg@10"],
        **kwargs,
    ) -> None:
        self.args = args
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = args.seq_len
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self.tokenizer.pad_token = "<unk>"
        if not self.tokenizer.eos_token:
            self.tokenizer.eos_token_id = 102
        self.tokenizer.sep_token = self.tokenizer.eos_token
        logger.info(f"pad={self.tokenizer.pad_token} sep={self.tokenizer.sep_token} eos={self.tokenizer.eos_token}")
        self.eval_metrics = EvalMetrics(eval_metrics, tokenizer)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        if model.config.num_labels == 1:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        args.label_names = ["labels"]
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        args.remove_unused_columns = False
        args.include_inputs_for_metrics = True
        train_processed = fmt.prepare_dataset(
            dataset=train_dataset, num_workers=args.dataloader_num_workers, num_negatives=args.num_negatives
        )
        if eval_dataset is not None:
            eval_processed = fmt.prepare_dataset(
                dataset=eval_dataset, num_workers=args.dataloader_num_workers, num_negatives=args.num_negatives
            )
        else:
            eval_processed = None
            args.evaluation_strategy = "no"

        super().__init__(
            args=args,
            model=model,
            compute_metrics=self.eval_metrics.compute_ce,
            train_dataset=train_processed,
            eval_dataset=eval_processed,
            data_collator=fmt.collate,
            tokenizer=tokenizer,
            **kwargs,
        )

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        model_predictions = model(**inputs, return_dict=True)
        logits = model_predictions["logits"]  # .view(-1)
        loss = self.loss(logits, inputs["labels"])
        if return_outputs:
            return loss, {"logits": logits}
        else:
            return loss
