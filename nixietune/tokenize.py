from abc import abstractmethod
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

    @abstractmethod
    def collate(self, items: List[Dict[str, Dict]]) -> Dict[str, Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def collect(self, inputs: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        pass

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

        queries = []
        docs = []
        lengths = []
        labels = []
        for q, p, n in zip(batch.get("query", []), batch.get("pos", []), batch.get("neg", [])):
            query_tokens = token_cache[q]
            for pos in p:
                doc = pos["doc"]
                doc_tokens = token_cache[doc]
                queries.append(query_tokens)
                docs.append(doc_tokens)
                lengths.append(query_tokens["input_ids"].size + doc_tokens["input_ids"].size)
                labels.append(float(pos.get("score", 1)))
            for neg in n:
                doc = neg["doc"]
                doc_tokens = token_cache[doc]
                queries.append(query_tokens)
                docs.append(doc_tokens)
                lengths.append(query_tokens["input_ids"].size + doc_tokens["input_ids"].size)
                labels.append(float(pos.get("score", 0)))
        result = {"query": queries, "doc": docs, "label": labels, "length": lengths}
        return result

    def collate(self, items: List[Dict[str, Any]]) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
        queries = [item["query"] for item in items]
        docs = [item["doc"] for item in items]
        padded_queries = self.pad(queries)
        padded_docs = self.pad(docs)
        result = {
            "query": padded_queries,
            "doc": padded_docs,
            "label": torch.tensor([item["label"] for item in items]),
        }
        return result

    def collect(self, items: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        result = []
        for column in ["query", "doc"]:
            result.append({"input_ids": items[column].input_ids, "attention_mask": items[column].attention_mask})
        return result
