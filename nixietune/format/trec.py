from datasets import Dataset, load_dataset
from typing import Optional, Dict, List
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass
from nixietune.log import setup_logging
import logging
import csv
import json
from pathlib import Path

setup_logging()
logger = logging.getLogger()


class TRECDatasetReader:
    def __init__(self, path: str, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        self.path = path
        self.tokenizer = tokenizer

    def corpus(
        self, subpath: str = "corpus.jsonl", max_length: int = 256, fields: List[str] = ["title", "text"]
    ) -> Dataset:
        return self._load_dict(subpath=subpath, max_length=max_length, fields=fields)

    def queries(self, subpath: str = "queries.jsonl", max_length=128) -> Dataset:
        return self._load_dict(subpath=subpath, max_length=max_length, fields=["text"])

    def qrels(self, subpath: str) -> Dataset:
        ds = load_dataset("csv", data_files={"train": f"{self.path}/{subpath}"}, split="train", sep="\t")
        return ds

    def join_query_doc_score(self, corpus: Dataset, queries: Dataset, qrels: Dataset) -> Dataset:
        joiner = QueryDocScoreJoiner(docs=corpus.to_dict(), queries=queries.to_dict())
        joined = qrels.map(function=joiner.join, batched=True)
        return joined.select_columns(["query", "passage", "label"])

    def _load_dict(self, subpath: str, max_length: int, fields: List[str]) -> Dataset:
        ds = load_dataset("json", data_files={"train": f"{self.path}/{subpath}"}, split="train")
        if self.tokenizer is not None:
            tok = TokenizerCallable(tokenizer=self.tokenizer, fields=fields, max_length=max_length)
            ds = ds.map(function=tok.tokenize_batch, batched=True)
        ds = ds.select_columns(["_id", "text"])
        return ds


class TRECDatasetWriter:
    @classmethod
    def save(self, ds: Dataset, path: str):
        data_dict = ds.to_dict()
        docs = {}
        queries = {}
        Path(f"{path}/qrels/").mkdir(parents=True, exist_ok=True)

        with open(f"{path}/qrels/train.tsv", "w") as qrel_file:
            qrel_csv = csv.writer(qrel_file, delimiter="\t")
            qrel_csv.writerow(["query-id", "corpus-id", "score"])
            for doc, query in zip(data_dict["text"], data_dict["query"]):
                if doc not in docs:
                    docs[doc] = len(docs)
                if query not in queries:
                    queries[query] = len(query)
                qrel_csv.writerow([queries[query], docs[doc], 1])

        with open(f"{path}/corpus.jsonl", "w") as corpus_file:
            for doc, id in docs.items():
                corpus_file.write(json.dumps({"_id": id, "text": doc}) + "\n")

        with open(f"{path}/queries.jsonl", "w") as queries_file:
            for q, id in queries.items():
                queries_file.write(json.dumps({"_id": id, "text": q}) + "\n")


class QueryDocScoreJoiner:
    def __init__(self, docs: Dict[str, List[str]], queries: Dict[str, List[str]]) -> None:
        def unwrap(data: Dict[str, List[str]]) -> Dict[str, str]:
            result = {}
            for id, text in zip(data["_id"], data["text"]):
                result[id] = text
            return result

        self.docs = unwrap(docs)
        self.queries = unwrap(queries)

    def join(self, batch: Dict[str, List[str]]) -> Dict[str, List[str]]:
        doc_texts = [self.docs[id] for id in batch["corpus-id"]]
        query_texts = [self.queries[id] for id in batch["query-id"]]
        scores = [float(s) for s in batch["score"]]
        return {"query": query_texts, "passage": doc_texts, "label": scores}


@dataclass
class TokenizerCallable:
    tokenizer: PreTrainedTokenizerBase
    fields: List[str]
    max_length: int

    def tokenize_batch(self, batch: Dict[str, List[str]]) -> Dict[str, List]:
        merged = []
        for row in zip(*[batch[field] for field in self.fields]):
            text = ""
            for col in row:
                text = f"{text} {col}" if col != "" else text
            merged.append(text)
        output = self.tokenizer(merged, padding=False, truncation=True, max_length=self.max_length)
        batch["text"] = output["input_ids"]
        return batch
