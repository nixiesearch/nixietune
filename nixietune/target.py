from abc import abstractmethod
from torch import nn
from nixietune.tokenize import DocTokenizer, QueryDocLabelTokenizer, QueryPosNegsTokenizer
from typing import List, Optional
from torch.nn.modules import Module
from sentence_transformers import SentenceTransformer, losses, util
from transformers import PreTrainedTokenizerBase


class Target:
    @abstractmethod
    def loss(self) -> nn.Module:
        pass

    @abstractmethod
    def process(self) -> DocTokenizer:
        pass


class CosineSimilarityTarget(Target):
    def __init__(self, model: SentenceTransformer, tokenizer: PreTrainedTokenizerBase) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def loss(self) -> Module:
        return losses.CosineSimilarityLoss(self.model)

    def process(self) -> DocTokenizer:
        return QueryDocLabelTokenizer(self.tokenizer)


class ContrastiveTarget(Target):
    def __init__(self, model: SentenceTransformer, tokenizer: PreTrainedTokenizerBase) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def loss(self) -> Module:
        return losses.ContrastiveLoss(self.model)

    def process(self) -> DocTokenizer:
        return QueryDocLabelTokenizer(self.tokenizer)


class InfoNCETarget(Target):
    def __init__(self, model: SentenceTransformer, tokenizer: PreTrainedTokenizerBase, num_negs: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.num_negs = num_negs

    def loss(self) -> Module:
        return losses.MultipleNegativesRankingLoss(self.model)

    def process(self) -> DocTokenizer:
        return QueryPosNegsTokenizer(self.tokenizer, self.num_negs)
