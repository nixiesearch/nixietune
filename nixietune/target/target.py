from abc import abstractmethod
from torch import nn
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase
from typing import Optional
from nixietune.format import Format


class Target:
    def __init__(
        self,
        model: SentenceTransformer,
        tokenizer: PreTrainedTokenizerBase,
        query_prefix: Optional[str],
        doc_prefix: Optional[str],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    @abstractmethod
    def loss(self) -> nn.Module:
        pass

    @abstractmethod
    def process(self) -> Format:
        pass
