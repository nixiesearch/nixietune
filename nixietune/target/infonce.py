from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase
from typing import Optional, Iterable, Dict
from nixietune.format import Format, QueryPosNegsFormat
from nixietune.target import Target
from torch import nn, Tensor
from info_nce import info_nce
import torch


class InfoNCELoss(nn.Module):
    def __init__(self, model: SentenceTransformer, temperature=0.05, reduction="mean", negative_mode="unpaired"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.model = model

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        queries = reps[0]
        positives = reps[1]
        if self.negative_mode == "paired":
            if len(reps) > 2:
                negatives = torch.stack(reps[2:], dim=1)
            else:
                raise ValueError("Cannot use paired InfoNCE loss with no negatives")
            return info_nce(
                queries,
                positives,
                negatives,
                temperature=self.temperature,
                reduction=self.reduction,
                negative_mode=self.negative_mode,
            )
        elif self.negative_mode == "unpaired":
            if len(reps) > 2:
                negatives = torch.cat(reps[2:])
            else:
                negatives = None
            return info_nce(
                queries,
                positives,
                negatives,
                temperature=self.temperature,
                reduction=self.reduction,
                negative_mode=self.negative_mode,
            )


class InfoNCETarget(Target):
    def __init__(
        self,
        model: SentenceTransformer,
        tokenizer: PreTrainedTokenizerBase,
        num_negs: int,
        query_prefix: Optional[str],
        doc_prefix: Optional[str],
        temperature: float,
        negative_mode: str,
    ) -> None:
        super().__init__(model, tokenizer, query_prefix, doc_prefix)
        self.num_negs = num_negs
        self.temperature = temperature
        self.negative_mode = negative_mode

    def loss(self) -> nn.Module:
        # return losses.MultipleNegativesRankingLoss(self.model)
        return InfoNCELoss(self.model, negative_mode=self.negative_mode, temperature=self.temperature)

    def process(self) -> Format:
        return QueryPosNegsFormat(self.tokenizer, self.query_prefix, self.doc_prefix, self.num_negs)
