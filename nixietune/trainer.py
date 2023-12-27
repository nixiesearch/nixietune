from transformers import PreTrainedModel
from sentence_transformers import SentenceTransformer
from transformers import TrainingArguments, Trainer
from typing import List, Dict, Any, Union, Tuple, Optional
import torch
from torch import nn
from dataclasses import dataclass, field
from datasets import Dataset
from nixietune.metrics import EvalMetrics
from nixietune.tokenize import QueryDocLabelTokenizer
from nixietune.target import CosineSimilarityTarget, ContrastiveTarget, InfoNCETarget
import logging
from transformers.tokenization_utils_base import BatchEncoding

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

    def forward(self, tensor, return_loss: bool = True):
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
        tokenizer = model.tokenizer
        tokenizer.model_max_length = args.seq_len
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        match args.target:
            case "cosine_similarity":
                self.target = CosineSimilarityTarget(model, tokenizer)
            case "contrastive":
                self.target = ContrastiveTarget(model, tokenizer)
            case "infonce":
                self.target = InfoNCETarget(model, tokenizer, 4)
        self.processor = self.target.process()
        self.loss = self.target.loss()
        self.loss.to(model.device)
        args.label_names = ["label"]
        train_processed = train_dataset.map(
            self.processor.tokenize,
            batched=True,
            batch_size=128,
            num_proc=args.dataloader_num_workers,
            remove_columns=["query", "pos", "neg"],
            desc="Tokenizing train dataset",
        )
        eval_processor = QueryDocLabelTokenizer(tokenizer)
        eval_processed = eval_dataset.map(
            eval_processor.tokenize,
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
            data_collator=self.collate,
            tokenizer=tokenizer,
            **kwargs,
        )

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: Dict[str, Dict[str, torch.Tensor]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        features = []
        for column in inputs["features"]:
            features.append({"input_ids": column.input_ids, "attention_mask": column.attention_mask})
        if self.target.label_name() is None:
            loss = self.loss(features, torch.Tensor())
        else:
            loss = self.loss(features, inputs[self.target.label_name()])
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

    def collate(self, items: List[Dict[str, Dict]]) -> Dict[str, Dict[str, torch.Tensor]]:
        features = []
        for item in items:
            for index, feature in enumerate(item["features"]):
                if index >= len(features):
                    features.extend([[]] * (index + 1 - len(features)))
                features[index].append(feature)

        padded_features = [self.pad(f) for f in features]
        result = {}
        result["features"] = padded_features
        if "label" in items[0]:
            result["label"] = torch.tensor([item["label"] for item in items])
        else:
            result["return_loss"] = True
        return result

    def pad(self, docs: List[Dict[str, Any]]) -> BatchEncoding:
        batch = BatchEncoding(
            data={
                "input_ids": [f["input_ids"] for f in docs],
                "attention_mask": [f["attention_mask"] for f in docs],
            }
        )
        return self.tokenizer.pad(batch, padding="longest", pad_to_multiple_of=8, return_tensors="pt")
