from datasets import Dataset, load_dataset, Sequence, Value, Features
from typing import Iterator, Optional, Dict, List, Any
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass
from nixietune.log import setup_logging
import logging
import csv
import json
from pathlib import Path
import os

setup_logging()
logger = logging.getLogger()


class TRECDataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        corpus: Dataset,
        queries: Dataset,
        qrels: Dataset,
        max_length: int = 526,
    ) -> None:
        def tokenize_dataset(ds: Dataset, fields: List[str] = ["text"]) -> tuple[Dataset, Dict[str, int]]:
            tok = TokenizerCallable(tokenizer=tokenizer, fields=fields, max_length=max_length)
            result = ds.map(function=tok.tokenize_batch, batched=True)
            result = result.select_columns(["_id", "text", "input_ids"])
            index = {key: index for index, key in enumerate(result["_id"])}
            return result, index

        self.corpus, self.corpus_index = tokenize_dataset(
            ds=corpus, fields=[field for field in corpus.column_names if field != "_id"]
        )
        self.queries, self.queries_index = tokenize_dataset(ds=queries)
        self.qrels = qrels
        logger.info(
            f"Loaded TREC dataset: corpus={len(self.corpus_index)} queries={len(self.queries_index)} qrels={len(qrels)}"
        )

    def as_tokenized_pairs(self) -> Dataset:
        def join(qrel: Dict[str, str]) -> Dict[str, Any]:
            qrel["query"] = self.queries[self.queries_index[qrel["query-id"]]]["input_ids"]
            qrel["doc"] = self.corpus[self.corpus_index[qrel["corpus-id"]]]["input_ids"]
            qrel["label"] = float(qrel["score"])
            return qrel

        result = self.qrels.map(function=join)
        schema = Features(
            {"query": Sequence(Value("int64")), "doc": Sequence(Value("int64")), "label": Value("double")}
        )
        return result.select_columns(["query", "doc", "label"]).cast(features=schema)

    def as_tokenized_triplets(self, threshold: float = 0.5) -> Dataset:
        qrels_pos: Dict[str, List[str]] = {}
        qrels_neg: Dict[str, List[str]] = {}
        for q, doc, label in zip(self.qrels["query-id"], self.qrels["corpus-id"], self.qrels["score"]):
            if float(label) > threshold:
                qrels_pos[q] = qrels_pos.get(q, []) + [doc]
            else:
                qrels_neg[q] = qrels_neg.get(q, []) + [doc]

        def generate():
            for q in self.qrels["query-id"]:
                query = self.queries[self.queries_index[q]]["input_ids"]
                pos = [self.corpus[self.corpus_index[doc]]["input_ids"] for doc in qrels_pos.get(q, [])]
                neg = [self.corpus[self.corpus_index[doc]]["input_ids"] for doc in qrels_neg.get(q, [])]
                yield {"query": query, "pos": pos, "neg": neg}

        schema = Features(
            {
                "query": Sequence(Value("int64")),
                "pos": Sequence(Sequence(Value("int64"))),
                "neg": Sequence(Sequence(Value("int64"))),
            }
        )
        return Dataset.from_generator(generator=generate, features=schema)

    @staticmethod
    def from_dir(
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        corpus_file: str = "corpus.jsonl",
        queries_file: str = "queries.jsonl",
        qrel_file: str = os.path.join("qrels", "train.tsv"),
    ):
        corpus = load_dataset("json", data_files={"train": os.path.join(path, corpus_file)}, split="train")
        queries = load_dataset("json", data_files={"train": os.path.join(path, queries_file)}, split="train")
        qrels = load_dataset("csv", data_files={"train": os.path.join(path, qrel_file)}, split="train", delimiter="\t")
        logger.info(f"Loading TREC dataset from local path {path}")
        logger.info(f"Corpus schema: {corpus.features}")
        logger.info(f"Query schema: {queries.features}")
        logger.info(f"QRel schema: {qrels.features}")
        return TRECDataset(tokenizer, corpus, queries, qrels)


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
        return {"_id": batch["_id"], "text": merged, "input_ids": output["input_ids"]}
