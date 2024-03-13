from transformers import Trainer, AutoModelForSequenceClassification, PreTrainedTokenizerBase
from nixietune.crossencoder.arguments import CrossEncoderArguments
from datasets import Dataset
from typing import List, Optional, Dict, Any, Union, Tuple
from nixietune.metrics import EvalMetrics
import torch
from transformers.tokenization_utils_base import BatchEncoding
from torch import nn
import random


class CrossEncoderTrainer(Trainer):
    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: PreTrainedTokenizerBase,
        args: CrossEncoderArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        eval_metrics: List[str] = ["ndcg@10"],
        **kwargs,
    ) -> None:
        self.args = args
        self.eval_metrics = EvalMetrics(eval_metrics, tokenizer)
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = args.seq_len
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self.loss = nn.BCEWithLogitsLoss()
        args.label_names = ["labels"]
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        args.remove_unused_columns = False
        args.include_inputs_for_metrics = True
        fmt = CrossEncoderDataset(self.tokenizer, args.seq_len)
        train_processed = fmt.prepare_dataset(
            dataset=train_dataset, num_workers=args.dataloader_num_workers, num_negatives=args.num_negatives
        )
        if eval_dataset is not None:
            eval_processed = fmt.prepare_dataset(
                dataset=eval_dataset, num_workers=args.dataloader_num_workers, num_negatives=None
            )
        else:
            eval_processed = None
            args.evaluation_strategy = "no"

        if args.max_steps == -1:
            batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size
            dataset_size = len(train_processed)
            args.max_steps = int(dataset_size / batch_size)
            print(f"dataset {dataset_size} batch {batch_size}")
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
        logits = model_predictions["logits"].view(-1)
        loss = self.loss(logits, inputs["labels"])
        if return_outputs:
            return loss, {"logits": logits}
        else:
            return loss


class CrossEncoderDataset:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def prepare_dataset(self, dataset: Dataset, num_negatives: Optional[int], num_workers: int = 1) -> Dataset:
        def cross_layout(batch: Dict[str, List]) -> Dict[str, List]:
            pairs = []
            labels = []
            for query, pos, negs, negscores in zip(batch["query"], batch["doc"], batch["neg"], batch["negscore"]):
                pairs.append((query, pos))
                labels.append(1.0)
                if len(negs) > 0:
                    if num_negatives:
                        neg_sample = random.choices(list(zip(negs, negscores)), k=num_negatives)
                        for neg, score in neg_sample:
                            pairs.append((query, neg))
                            labels.append(score)
                    else:
                        for neg, score in zip(negs, negscores):
                            pairs.append((query, neg))
                            labels.append(score)

            result = self.tokenizer(pairs, padding=False, truncation=True, max_length=self.max_len)
            result["labels"] = labels
            return result

        return dataset.map(
            function=cross_layout,
            batched=True,
            desc="tokenizing",
            remove_columns=["query", "doc", "neg", "negscore"],
            num_proc=num_workers,
        )

    def collate(self, items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        data: Dict[str, List] = {}
        for item in items:
            for key, value in item.items():
                values = data.get(key)
                if values is not None:
                    values.append(value)
                else:
                    data[key] = [value]
        batch = BatchEncoding(data=data)
        encoded = self.tokenizer.pad(batch, padding="longest", pad_to_multiple_of=8, return_tensors="pt")
        encoded["labels"] = torch.tensor([input["labels"] for input in items])
        return encoded
