from abc import abstractmethod
from transformers import PreTrainedTokenizerBase
from typing import Dict, List, Optional
import numpy as np
from datasets import Features


class Format:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        query_prefix: Optional[str],
        doc_prefix: Optional[str],
    ) -> None:
        self.tokenizer = tokenizer
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    @abstractmethod
    def tokenize(self, batch: Dict[str, List]) -> Dict[str, List]:
        pass

    @abstractmethod
    def schema(self, dtype: str) -> Features:
        pass

    def make_tokenized_cache(self, batch: Dict[str, List]) -> Dict[str, Dict[str, np.ndarray]]:
        docs: List[str] = []
        for q, p, n in zip(batch["query"], batch.get("positive", []), batch.get("negative", [])):
            docs.append(self.format_query(q))
            for pos in p:
                docs.append(self.format_doc(pos))
            for neg in n:
                docs.append(self.format_doc(neg))
        tokens = self.tokenizer(docs, padding=False, truncation=True, return_tensors="np")
        token_cache: Dict[str, Dict] = {}
        for doc, input_ids, attention_mask in zip(docs, tokens.input_ids, tokens.attention_mask):
            token_cache[doc] = {
                "input_ids": input_ids.astype("int32"),
                "attention_mask": attention_mask.astype("int8"),
            }
        return token_cache

    def format_doc(self, doc: str) -> str:
        if self.doc_prefix is None:
            return doc
        else:
            return self.doc_prefix + doc

    def format_query(self, query: str) -> str:
        if self.query_prefix is None:
            return query
        else:
            return self.query_prefix + query
