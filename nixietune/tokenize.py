from abc import abstractmethod
from torch._tensor import Tensor
from transformers import PreTrainedTokenizerBase
from typing import Dict, List, Any, Union
import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding


class DocTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    def tokenize(self, batch: Dict[str, List]) -> Dict[str, List]:
        pass

    def collate(self, items: List[Dict[str, Dict]]) -> Dict[str, Dict[str, Tensor]]:
        features = [[]] * (self.neg_count + 2)
        for item in items:
            for index, feature in enumerate(item["features"]):
                features[index].append(feature)

        padded_features = [self.pad(f) for f in features]
        result = {"features": padded_features, "return_loss": True}
        return result

    def make_tokenized_cache(self, batch: Dict[str, List]) -> Dict[str, Dict[str, np.ndarray]]:
        docs: List[str] = []
        for q, p, n in zip(batch["query"], batch.get("pos", []), batch.get("neg", [])):
            docs.append(q)
            for pos in p:
                docs.append(pos["doc"])
            for neg in n:
                docs.append(neg["doc"])
        tokens = self.tokenizer(docs, padding=False, truncation=True, return_tensors="np")
        token_cache: Dict[str, Dict] = {}
        for doc, input_ids, attention_mask in zip(docs, tokens.input_ids, tokens.attention_mask):
            token_cache[doc] = {
                "input_ids": input_ids.astype("int32"),
                "attention_mask": attention_mask.astype("int8"),
            }
        return token_cache

    def pad(self, docs: List[Dict[str, Any]]) -> BatchEncoding:
        batch = BatchEncoding(
            data={
                "input_ids": [f["input_ids"] for f in docs],
                "attention_mask": [f["attention_mask"] for f in docs],
            }
        )
        return self.tokenizer.pad(batch, padding="longest", pad_to_multiple_of=8, return_tensors="pt")


class QueryDocLabelTokenizer(DocTokenizer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, batch: Dict[str, List]) -> Dict[str, List]:
        token_cache = self.make_tokenized_cache(batch)

        features = []
        lengths = []
        labels = []
        for q, p, n in zip(batch.get("query", []), batch.get("pos", []), batch.get("neg", [])):
            query_tokens = token_cache[q]
            for pos in p:
                doc = pos["doc"]
                doc_tokens = token_cache[doc]
                features.append([query_tokens, doc_tokens])
                lengths.append(query_tokens["input_ids"].size + doc_tokens["input_ids"].size)
                labels.append(float(pos.get("score", 1)))
            for neg in n:
                doc = neg["doc"]
                doc_tokens = token_cache[doc]
                features.append([query_tokens, doc_tokens])
                lengths.append(query_tokens["input_ids"].size + doc_tokens["input_ids"].size)
                labels.append(float(pos.get("score", 0)))
        result = {"features": features, "label": labels, "length": lengths}
        return result


class QueryPosNegsTokenizer(DocTokenizer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, neg_count: int) -> None:
        self.tokenizer = tokenizer
        self.neg_count = neg_count

    def tokenize(self, batch: Dict[str, List]) -> Dict[str, List]:
        token_cache = self.make_tokenized_cache(batch)

        docs = []
        lengths = []
        for q, p, n in zip(batch.get("query", []), batch.get("pos", []), batch.get("neg", [])):
            if len(n) >= self.neg_count:
                query_tokens = token_cache[q]
                neg_tokens = []
                length = query_tokens["input_ids"].size
                for neg in n[:3]:
                    t = token_cache[neg["doc"]]
                    length += t["input_ids"].size
                    neg_tokens.append(t)
                for pos in p:
                    pos_tokens = token_cache[pos["doc"]]
                    docs.append([query_tokens, pos_tokens] + neg_tokens)
                    lengths.append(pos_tokens["input_ids"].size + length)
        result = {"features": docs, "length": lengths}
        return result
