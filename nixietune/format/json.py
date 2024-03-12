from datasets import Dataset, load_dataset, Features, Value, Sequence
from typing import Dict, List
import re
import os
from huggingface_hub.utils import validate_repo_id
import logging

logger = logging.getLogger()


class JSONDataset:
    @staticmethod
    def load(path: str, split: str, num_workers: int = 4) -> Dataset:
        if os.path.exists(path):
            if os.path.isdir(path):
                return JSONDataset.from_dir(path, split=split, num_workers=num_workers)
            else:
                return JSONDataset.from_file(path, split=split, num_workers=num_workers)
        else:
            validate_repo_id(path)
            dataset = load_dataset(path, split=split)
            return JSONDataset.from_dataset(dataset, num_workers=num_workers)

    @staticmethod
    def from_dir(path: str, split: str, num_workers: int = 4) -> Dataset:
        logger.info(f"Loading dataset from dir {path}")
        raw_split = re.sub("\[.*?\]", "", split)
        ds = load_dataset("json", data_dir={raw_split: path}, split=split)
        return JSONDataset.from_dataset(ds, num_workers)

    @staticmethod
    def from_file(path: str, split: str, num_workers: int = 4) -> Dataset:
        logger.info(f"Loading dataset from file {path}")
        raw_split = re.sub("\[.*?\]", "", split)
        ds = load_dataset("json", data_files={raw_split: path}, split=split)
        return JSONDataset.from_dataset(ds, num_workers)

    @staticmethod
    def from_dataset(ds: Dataset, num_workers: int = 4) -> Dataset:
        schema = Features(
            {
                "query": Value("string"),
                "doc": Value("string"),
                "neg": Sequence(Value("string")),
                "negscore": Sequence(Value("float")),
            }
        )
        return ds.map(
            function=JSONDataset.process_batch,
            batched=True,
            num_proc=num_workers,
            desc="Loading dataset",
            features=schema,
        )

    @staticmethod
    def process_batch(batch: Dict[str, List]) -> Dict[str, List]:
        if "query" not in batch and "doc" not in batch:
            raise Exception("dataset expected to have query and/or doc fields")
        else:
            batch_size = max(len(batch.get("query", [])), len(batch.get("doc", [])))

        queries = batch.get("query", [None] * batch_size)
        doc = batch.get("doc", [None] * batch_size)
        neg = []
        if "neg" in batch:
            for n in batch["neg"]:
                if n is not None:
                    neg.append(n)
                else:
                    neg.append([])
        else:
            neg = [[]] * batch_size
        negscore = []
        if "negscore" in batch:
            for n in batch["negscore"]:
                if n is not None:
                    negscore.append(n)
                else:
                    negscore.append([])
        else:
            if "neg" in batch:
                for negs in batch["neg"]:
                    negscore.append([0] * len(negs))
            else:
                negscore.append([])
        return {"query": queries, "doc": doc, "neg": neg, "negscore": negscore}
