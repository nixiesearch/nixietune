from sentence_transformers import SentenceTransformer, losses
from transformers import PreTrainedTokenizerBase
from typing import Optional
from nixietune.format import Format, QueryDocLabelFormat
from nixietune.target import Target
from torch import nn


class ContrastiveTarget(Target):
    def __init__(
        self,
        model: SentenceTransformer,
        tokenizer: PreTrainedTokenizerBase,
        query_prefix: Optional[str],
        doc_prefix: Optional[str],
    ) -> None:
        super().__init__(model, tokenizer, query_prefix, doc_prefix)

    def loss(self) -> nn.Module:
        return losses.ContrastiveLoss(self.model)

    def process(self) -> Format:
        return QueryDocLabelFormat(self.tokenizer, self.query_prefix, self.doc_prefix)
