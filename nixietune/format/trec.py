from datasets import Dataset, load_dataset, Sequence, Value, Features, DatasetDict
from typing import Optional, Dict, List, Any
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass
from nixietune.log import setup_logging
import logging
import os
import pandas as pd

setup_logging()
logger = logging.getLogger()


class TRECDataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        corpus: Dataset,
        queries: Dataset,
        qrels: DatasetDict,
        max_length: int = 512,
        corpus_fields: List[str] = ["text"],
        doc_prefix: Optional[str] = None,
        query_prefix: Optional[str] = None,
    ) -> None:
        self.corpus = (
            TRECDataset.tokenize_dataset(
                ds=corpus,
                tokenizer=tokenizer,
                max_length=max_length,
                fields=corpus_fields,
                prefix=doc_prefix,
                desc="Tokenizing corpus",
            )
            .to_pandas()
            .set_index("_id")
        )
        self.queries = (
            TRECDataset.tokenize_dataset(
                ds=queries,
                tokenizer=tokenizer,
                max_length=max_length,
                prefix=query_prefix,
                desc="Tokenizing queries",
            )
            .to_pandas()
            .set_index("_id")
        )
        self.qrels = qrels
        qrel_stats = {split: len(qrel) for split, qrel in qrels.items()}
        logger.info(f"Loaded TREC dataset: corpus={len(self.corpus)} queries={len(self.queries)} qrels={qrel_stats}")

    @staticmethod
    def tokenize_dataset(
        ds: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        fields: List[str] = ["text"],
        prefix: Optional[str] = None,
        desc: str = "",
    ) -> Dataset:
        print(f"Tokenization params: max_len={max_length} fields={fields} prefix={prefix}")
        tok = TokenizerCallable(tokenizer=tokenizer, fields=fields, max_length=max_length, prefix=prefix)
        schema = Features({"_id": Value("string"), "text": Value("string"), "input_ids": Sequence(Value("int64"))})
        result = ds.map(function=tok.tokenize_batch, batched=True, desc=desc, num_proc=6)
        result = result.select_columns(["_id", "text", "input_ids"])
        result = result.cast(schema)
        return result

    def load_split(self, split: str) -> Dataset:
        qrels = self.qrels[split].to_pandas().set_index("query-id")
        grouped = qrels.groupby("query-id").agg({"corpus-id": list, "score": list})
        schema = Features(
            {
                "query": Sequence(Value("int32")),
                "docs": [Sequence(Value("int32"))],
                "scores": Sequence(Value("double")),
            }
        )
        joiner = CorpusJoinCallable(self.queries, self.corpus)
        ds = Dataset.from_pandas(grouped).map(
            function=joiner.join_corpus,
            batched=True,
            batch_size=4,
            desc="Joining qrels with tokenized corpus",
            features=schema,
            remove_columns=["query-id", "corpus-id", "score"],
        )
        return ds

    @staticmethod
    def from_dir(
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        corpus_file: str = "corpus.jsonl",
        queries_file: str = "queries.jsonl",
        qrel_splits: List[str] = ["train"],
        max_length: int = 512,
    ):
        corpus = load_dataset("json", data_files={"train": os.path.join(path, corpus_file)}, split="train")
        queries = load_dataset("json", data_files={"train": os.path.join(path, queries_file)}, split="train")
        qrels_full_path = {split: os.path.join(path, "qrels", split + ".tsv") for split in qrel_splits}
        qrels_schema = Features({"query-id": Value("string"), "corpus-id": Value("string"), "score": Value("double")})
        qrels = load_dataset("csv", data_files=qrels_full_path, delimiter="\t", features=qrels_schema)
        logger.info(f"Loading TREC dataset from local path {path}")
        logger.info(f"Corpus schema: {corpus.features}")
        logger.info(f"Query schema: {queries.features}")
        logger.info(f"QRel schema: {[qrel.features for split, qrel in qrels.items()]}")
        return TRECDataset(tokenizer, corpus, queries, qrels, max_length=max_length)

    @staticmethod
    def corpus_from_dir(
        path: str,
        tokenizer: PreTrainedTokenizerBase,
        corpus_file: str = "corpus.jsonl",
        max_length: int = 256,
        fields: List[str] = ["text"],
    ) -> pd.DataFrame:
        corpus_ds = load_dataset("json", data_files={"train": os.path.join(path, corpus_file)}, split="train")
        corpus = TRECDataset.tokenize_dataset(
            ds=corpus_ds,
            tokenizer=tokenizer,
            max_length=max_length,
            fields=fields,
        )
        return corpus


@dataclass
class TokenizerCallable:
    tokenizer: PreTrainedTokenizerBase
    fields: List[str]
    max_length: int
    prefix: Optional[str] = None

    def tokenize_batch(self, batch: Dict[str, List[str]]) -> Dict[str, List]:
        merged = []
        for row in zip(*[batch[field] for field in self.fields]):
            text = ""
            for col in row:
                text = f"{text} {col}" if col != "" else text
            if self.prefix is not None:
                merged.append(self.prefix + text)
            else:
                merged.append(text)
        output = self.tokenizer(merged, padding=False, truncation=True, max_length=self.max_length)
        return {"_id": batch["_id"], "text": merged, "input_ids": output["input_ids"]}


@dataclass
class CorpusJoinCallable:
    queries: pd.DataFrame
    corpus: pd.DataFrame

    def join_corpus(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        queries = []
        docs = []
        labels = []
        for qid, docids, scores in zip(batch["query-id"], batch["corpus-id"], batch["score"]):
            queries.append(self.queries.loc[qid, "input_ids"])
            docs.append([self.corpus.loc[docid, "input_ids"] for docid in docids])
            labels.append(scores)
        result = {"query": queries, "docs": docs, "scores": labels}
        return result

    # don't pickle the queries/corpus because it's super slow
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["queries"]
        del state["corpus"]
        return state
