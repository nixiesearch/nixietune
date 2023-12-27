from transformers import PreTrainedModel
from sentence_transformers import SentenceTransformer
from transformers import TrainingArguments, Trainer
from typing import List, Dict, Any, Union, Tuple, Optional
import torch
from torch import nn
from dataclasses import dataclass, field
from datasets import Dataset
from nixietune.metrics import EvalMetrics
from torch.utils.data import DataLoader
from transformers.trainer_utils import seed_worker
from nixietune.target import CosineSimilarityTarget, ContrastiveTarget
import logging

logger = logging.getLogger()


@dataclass
class BiencoderTrainingArguments(TrainingArguments):
    seq_len: int = field(
        default=128,
        metadata={"help": "Max sequence length in tokens."},
    )

    target: str = field(
        default="cosine_similarity", metadata={"help": "Optimization target: cosine_similarity/contrastive"}
    )


class BiencoderModel(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, model: nn.Sequential):
        super().__init__(model[0].auto_model.config)
        self.model = model

    def forward(self, tensor):
        return self.model.forward(tensor)


class BiencoderTrainer(Trainer):
    def __init__(
        self,
        model: SentenceTransformer,
        args: BiencoderTrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        eval_metrics: List[str] = ["ndcg@10"],
        **kwargs,
    ) -> None:
        self.eval_metrics = EvalMetrics(eval_metrics)
        self.tokenizer = model.tokenizer
        self.tokenizer.model_max_length = args.seq_len
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        match args.target:
            case "cosine_similarity":
                self.target = CosineSimilarityTarget(model, self.tokenizer)
            case "contrastive":
                self.target = ContrastiveTarget(model, self.tokenizer)
        self.processor = self.target.process()
        self.loss = self.target.loss()
        self.loss.to(model.device)
        train_processed = train_dataset.map(
            self.processor.tokenize,
            batched=True,
            batch_size=128,
            num_proc=args.dataloader_num_workers,
            remove_columns=["query", "pos", "neg"],
            desc="Tokenizing train dataset",
        )
        eval_processed = eval_dataset.map(
            self.processor.tokenize,
            batched=True,
            batch_size=128,
            num_proc=args.dataloader_num_workers,
            remove_columns=["query", "pos", "neg"],
            desc="Tokenizing test dataset",
        )
        super().__init__(
            args=args,
            model=BiencoderModel(model),
            compute_metrics=self.eval_metrics.compute,
            preprocess_logits_for_metrics=self.move_to_cpu,
            train_dataset=train_processed,
            eval_dataset=eval_processed,
            data_collator=self.processor.collate,
            **kwargs,
        )

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: Dict[str, Dict[str, torch.Tensor]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        features = self.processor.collect(inputs)
        loss = self.loss(features, inputs["label"])
        if return_outputs:
            output = [torch.Tensor()] + features
            return loss, output
        return loss

    def move_to_cpu(self, logits: List[Dict[str, torch.Tensor]], labels: torch.Tensor) -> List[torch.Tensor]:
        result = []
        for f in logits:
            se = f["sentence_embedding"]
            result.append({"sentence_embedding": se.clone()})
        return result
