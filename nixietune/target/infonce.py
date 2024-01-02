from sentence_transformers import SentenceTransformer, losses
from transformers import PreTrainedTokenizerBase
from typing import Optional
from nixietune.format import Format, QueryPosNegsFormat
from nixietune.target import Target
from torch import nn


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
        return losses.MultipleNegativesRankingLoss(self.model)

    def process(self) -> Format:
        return QueryPosNegsFormat(self.tokenizer, self.query_prefix, self.doc_prefix, self.num_negs)
