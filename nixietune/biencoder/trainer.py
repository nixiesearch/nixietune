from transformers import PreTrainedModel, PreTrainedTokenizerBase
from sentence_transformers import SentenceTransformer, losses
from transformers import Trainer
from typing import List, Dict, Any, Union, Tuple, Optional
import torch
from torch import nn

from datasets import Dataset
from nixietune.metrics import EvalMetrics
from nixietune.biencoder.arguments import BiencoderTrainingArguments
import logging
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
import numpy as np
from itertools import islice
from nixietune.format.trec import TRECDataset
from nixietune.biencoder.layout import QueryDocLabelLayout, Layout, QueryPosNegsLayout
from nixietune.target.infonce import InfoNCELoss

logger = logging.getLogger()


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
        tokenizer: PreTrainedTokenizerBase,
        args: BiencoderTrainingArguments,
        train_dataset: TRECDataset,
        train_split: str,
        eval_dataset: Optional[TRECDataset],
        eval_split: Optional[str],
        eval_metrics: List[str] = ["ndcg@10"],
        **kwargs,
    ) -> None:
        self.eval_metrics = EvalMetrics(eval_metrics)
        tokenizer = model.tokenizer
        tokenizer.model_max_length = args.seq_len
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        match args.target:
            case "cosine_similarity":
                self.loss = losses.CosineSimilarityLoss(model)
                self.format = QueryDocLabelLayout()
            case "infonce":
                self.loss = InfoNCELoss(
                    model, negative_mode=args.infonce_negative_mode, temperature=args.infonce_temperature
                )
                self.format = QueryPosNegsLayout(num_negatives=args.num_negatives)
            # case "contrastive":
            #     self.target = ContrastiveTarget(model, tokenizer, args.query_prefix, args.document_prefix)
            # case "mixed":
            #     self.target = MixedTarget(
            #         model,
            #         tokenizer,
            #         num_negs=args.num_negatives,
            #         query_prefix=args.query_prefix,
            #         doc_prefix=args.document_prefix,
            #     )
            # case "infonce":
            #     self.target = InfoNCETarget(
            #         model,
            #         tokenizer,
            #         num_negs=args.num_negatives,
            #         query_prefix=args.query_prefix,
            #         doc_prefix=args.document_prefix,
            #         temperature=args.infonce_temperature,
            #         negative_mode=args.infonce_negative_mode,
            #     )
            # case "triplet":
            #     self.target = TripletTarget(
            #         model, tokenizer, args.query_prefix, args.document_prefix, args.num_negatives, args.triplet_margin
            #     )
        self.loss.to(model.device)
        args.label_names = ["label"]
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        args.remove_unused_columns = False
        # self.print_raw_stats(train_dataset)
        train_processed = self.prepare_dataset(dataset=train_dataset.load_split(train_split), format=self.format)
        # self.print_tokenized_stats(train_processed)
        if eval_dataset is not None:
            self.eval_loss = losses.CosineSimilarityLoss(model)
            self.eval_format = QueryDocLabelLayout()
            eval_processed = self.prepare_dataset(dataset=eval_dataset.load_split(eval_split), format=self.eval_format)
            self.eval_loss.to(model.device)
        else:
            eval_processed = None
            args.evaluation_strategy = "no"
        bi_model = BiencoderModel(model)
        bi_model.warnings_issued["estimate_tokens"] = True
        if args.max_steps == -1:
            batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size
            dataset_size = len(train_processed)
            args.max_steps = int(dataset_size / batch_size)
            print(f"dataset {dataset_size} batch {batch_size}")

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

    def prepare_dataset(self, dataset: Dataset, format: Layout) -> Dataset:
        processed = dataset.map(
            function=format.unwrap,
            batched=True,
            batch_size=128,
            remove_columns=["query", "docs", "scores"],
            desc=format.desc(),
            features=format.schema(),
        )
        return processed

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
        # else:
        #    result["return_loss"] = True
        return result

    def pad(self, docs: List[Dict[str, Any]]) -> BatchEncoding:
        batch = BatchEncoding(
            data={
                "input_ids": docs,
                "attention_mask": [[1] * len(f) for f in docs],
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
