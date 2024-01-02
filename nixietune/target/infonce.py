from sentence_transformers import SentenceTransformer, losses
from transformers import PreTrainedTokenizerBase
from typing import Optional, Iterable, Dict
from nixietune.format import Format, QueryPosNegsFormat
from nixietune.target import Target
from torch import nn, Tensor
from info_nce import info_nce
import torch


class InfoNCELoss(nn.Module):
    def __init__(self, model: SentenceTransformer, temperature=0.003, reduction="mean", negative_mode="unpaired"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.model = model

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        queries = reps[0]
        positives = reps[1]
        negatives = torch.stack(reps[2:], dim=1)
        return info_nce(
            queries,
            positives,
            negatives,
            temperature=self.temperature,
            reduction=self.reduction,
            negative_mode="paired",
        )


class InfoNCETarget(Target):
    def __init__(
        self,
        model: SentenceTransformer,
        tokenizer: PreTrainedTokenizerBase,
        num_negs: int,
        query_prefix: Optional[str],
        doc_prefix: Optional[str],
    ) -> None:
        super().__init__(model, tokenizer, query_prefix, doc_prefix)
        self.num_negs = num_negs

    def loss(self) -> nn.Module:
        return InfoNCELoss(self.model)

    def process(self) -> Format:
        return QueryPosNegsFormat(self.tokenizer, self.query_prefix, self.doc_prefix, self.num_negs)
