from sentence_transformers import SentenceTransformer, losses
from transformers import PreTrainedTokenizerBase
from typing import Optional
from nixietune.format import Format, TripletFormat
from nixietune.target import Target
from torch import nn


class TripletTarget(Target):
    def __init__(
        self,
        model: SentenceTransformer,
        tokenizer: PreTrainedTokenizerBase,
        query_prefix: Optional[str],
        doc_prefix: Optional[str],
        num_negs: int,
        margin: float,
    ) -> None:
        super().__init__(model, tokenizer, query_prefix, doc_prefix)
        self.num_negs = num_negs
        self.margin = margin

    def loss(self) -> nn.Module:
        return losses.TripletLoss(
            self.model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=self.margin
        )

    def process(self) -> Format:
        return TripletFormat(self.tokenizer, self.query_prefix, self.doc_prefix, neg_count=self.num_negs)
