from transformers import PreTrainedModel
from sentence_transformers import SentenceTransformer
from transformers import TrainingArguments, Trainer
from typing import List, Dict, Any, Union, Tuple, Optional
import torch
from torch import nn
from dataclasses import dataclass, field
from datasets import Dataset
from nixietune.metrics import EvalMetrics
from nixietune.target import CosineSimilarityTarget, ContrastiveTarget, InfoNCETarget, TripletTarget
import logging
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
import numpy as np
from itertools import islice

logger = logging.getLogger()


@dataclass
class BiencoderTrainingArguments(TrainingArguments):
    seq_len: int = field(
        default=128,
        metadata={"help": "Max sequence length in tokens."},
    )

    target: str = field(
        default="infonce", metadata={"help": "Optimization target: cosine_similarity/contrastive/infonce"}
    )

    num_negatives: int = field(default=4, metadata={"help": "Number of negatives to use for InfoNCE/Triplet loss"})

    triplet_margin: float = field(default=5, metadata={"help": "Margin value for Triplet loss"})

    query_prefix: Optional[None] = field(
        default=None, metadata={"help": "Prefix for all queries. Used for asymmetrical models like E5."}
    )

    document_prefix: Optional[None] = field(
        default=None, metadata={"help": "Prefix for all documents. Used for asymmetrical models like E5."}
    )


class BiencoderModel(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, model: nn.Sequential):
        super().__init__(model[0].auto_model.config)
        self.model = model

    def forward(self, tensor, return_loss: bool = False):
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
                self.target = CosineSimilarityTarget(model, tokenizer, args.query_prefix, args.document_prefix)
            case "contrastive":
                self.target = ContrastiveTarget(model, tokenizer, args.query_prefix, args.document_prefix)
            case "infonce":
                self.target = InfoNCETarget(
                    model, tokenizer, args.num_negatives, args.query_prefix, args.document_prefix
                )
            case "triplet":
                self.target = TripletTarget(
                    model, tokenizer, args.query_prefix, args.document_prefix, args.num_negatives, args.triplet_margin
                )
        self.eval_target = CosineSimilarityTarget(model, tokenizer, args.query_prefix, args.document_prefix)
        self.processor = self.target.process()
        self.eval_processor = self.eval_target.process()
        self.loss = self.target.loss()
        self.loss.to(model.device)
        self.eval_loss = self.eval_target.loss()
        self.eval_loss.to(model.device)
        args.label_names = ["label"]
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        args.remove_unused_columns = False
        # self.print_raw_stats(train_dataset)
        if len(tokenizer) < 65536:
            dtype = "uint16"
        else:
            dtype = "uint32"
        logger.info(f"Tokenizer vocab_size={len(tokenizer)}, using {dtype} for token encoding on disk cache")
        train_processed = train_dataset.map(
            self.processor.tokenize,
            batched=True,
            batch_size=128,
            num_proc=args.dataloader_num_workers,
            remove_columns=["query", "positive", "negative"],
            desc="Tokenizing train dataset",
            features=self.processor.schema(dtype),
        )
        # self.print_tokenized_stats(train_processed)
        eval_processed = eval_dataset.map(
            self.eval_processor.tokenize,
            batched=True,
            batch_size=128,
            num_proc=args.dataloader_num_workers,
            remove_columns=["query", "positive", "negative"],
            desc="Tokenizing test dataset",
            features=self.eval_processor.schema(dtype),
        )
        bi_model = BiencoderModel(model)
        bi_model.warnings_issued["estimate_tokens"] = True
        super().__init__(
            args=args,
            model=bi_model,
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
        if return_outputs is True:
            target_loss = self.eval_loss
        else:
            target_loss = self.loss
        if "label" in inputs:
            loss = target_loss(features, inputs["label"])
        else:
            loss = target_loss(features, torch.Tensor())
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

    def print_raw_stats(self, dataset: Dataset, samples: int = 5000) -> None:
        positives_per_query = []
        negatives_per_query = []
        for row in tqdm(islice(dataset, samples), desc="Collecting raw stats", total=samples):
            pos = len(row["positive"])
            positives_per_query.append(pos)
            neg = len(row["negative"])
            negatives_per_query.append(neg)
        ppq = np.percentile(positives_per_query, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        npq = np.percentile(negatives_per_query, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        logger.info(f"Negatives per query: {npq}")
        logger.info(f"Positives per query: {ppq}")

    def print_tokenized_stats(self, dataset: Dataset, samples: int = 5000) -> None:
        query_tokens = []
        doc_tokens = []
        for row in tqdm(islice(dataset, samples), desc="Collecting token stats", total=samples):
            features = row["features"]
            query, positive, *docs = features
            query_tokens.append(len(query["input_ids"]))
            doc_tokens.append(len(positive["input_ids"]))
            [doc_tokens.append(len(neg["input_ids"])) for neg in docs]
        qtp = np.percentile(query_tokens, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], method="nearest")
        dtp = np.percentile(doc_tokens, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], method="nearest")
        logger.info(f"Query tokens:    {qtp}")
        logger.info(f"Document tokens: {dtp}")
