from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase
from typing import Optional, Iterable, Dict
from nixietune.format import Format, QueryPosNegsFormat
from nixietune.target import Target
from torch import nn, Tensor
from info_nce import info_nce
import torch


class MixedLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, temperature=0.05, nce_weight: int = 1.0, cosine_weight: int = 1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.nce_weight = nce_weight
        self.cosine_weight = cosine_weight

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        queries = reps[0]
        positives = reps[1]
        if len(reps) > 2:
            negatives = torch.cat(reps[2:])
            nce = info_nce(queries, positives, negatives, temperature=self.temperature)
            cos = info_nce(queries, positives, temperature=self.temperature)
            return self.nce_weight * nce + self.cosine_weight * cos
        else:
            raise ValueError("Cannot use mixed loss with no negatives")


class MixedTarget(Target):
    def __init__(
        self,
        model: SentenceTransformer,
        tokenizer: PreTrainedTokenizerBase,
        num_negs: int,
        query_prefix: Optional[str],
        doc_prefix: Optional[str],
        nce_weight: float = 1.0,
        cosine_weight: float = 1.0,
        temperature: float = 0.05,
    ) -> None:
        super().__init__(model, tokenizer, query_prefix, doc_prefix)
        self.num_negs = num_negs
        self.nce_weight = nce_weight
        self.cosine_weight = cosine_weight
        self.temperature = temperature

    def loss(self) -> nn.Module:
        return MixedLoss(
            self.model,
            nce_weight=self.nce_weight,
            cosine_weight=self.cosine_weight,
            temperature=self.temperature,
        )

    def process(self) -> Format:
        return QueryPosNegsFormat(self.tokenizer, self.query_prefix, self.doc_prefix, self.num_negs)
