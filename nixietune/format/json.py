from datasets import Dataset, load_dataset, Features, Value, Sequence
from transformers import PreTrainedTokenizerBase
from typing import Dict, List
import logging
import os
import re
from huggingface_hub.utils import validate_repo_id

logger = logging.getLogger()


class JSONDataset:
    @staticmethod
    def load(path: str, tok: PreTrainedTokenizerBase, split: str, max_len: int, num_workers: int = 4) -> Dataset:
        if os.path.exists(path):
            if os.path.isdir(path):
                return JSONDataset.from_dir(path, tokenizer=tok, split=split, max_len=max_len, num_workers=num_workers)
            else:
                return JSONDataset.from_file(
                    path, tokenizer=tok, split=split, max_len=max_len, num_workers=num_workers
                )
        else:
            validate_repo_id(path)
            dataset = load_dataset(path, split=split)
            return JSONDataset.from_dataset(dataset, tokenizer=tok, max_len=max_len, num_workers=num_workers)

    @staticmethod
    def from_dataset(ds: Dataset, tokenizer: PreTrainedTokenizerBase, max_len: int, num_workers: int = 4) -> Dataset:
        return JSONDataset._tokenize(ds, tokenizer, max_len, num_workers)

    @staticmethod
    def from_file(
        path: str, tokenizer: PreTrainedTokenizerBase, max_len: int, split: str, num_workers: int = 4
    ) -> Dataset:
        raw_split = re.sub("\[.*?\]", "", split)
        source = load_dataset("json", data_files={raw_split: path}, split=split)
        return JSONDataset._tokenize(source, tokenizer, max_len, num_workers)

    @staticmethod
    def from_dir(
        path: str, tokenizer: PreTrainedTokenizerBase, max_len: int, split: str, num_workers: int = 4
    ) -> Dataset:
        raw_split = re.sub("\[.*?\]", "", split)
        source = load_dataset("json", data_dir={raw_split: path}, split=split)
        return JSONDataset._tokenize(source, tokenizer, max_len, num_workers)

    @staticmethod
    def _tokenize(data: Dataset, tokenizer: PreTrainedTokenizerBase, max_len: int, num_workers: int) -> Dataset:
        def process_batch(batch: Dict[str, List]) -> Dict[str, List]:
            if "query" in batch:
                query = tokenizer(batch["query"], padding=False, truncation=True, max_length=max_len)["input_ids"]
            else:
                query = [None] * len(batch["pos"])
            positive = tokenizer(batch["pos"], padding=False, truncation=True, max_length=max_len)["input_ids"]

            negatives = []
            if "neg" in batch:
                all_negatives = [neg for neglist in batch["neg"] for neg in neglist]
                if len(all_negatives) > 0:
                    negatives_tokenized = tokenizer(all_negatives, padding=False, truncation=True, max_length=max_len)[
                        "input_ids"
                    ]
                    neg_dict = {text: tok for text, tok in zip(all_negatives, negatives_tokenized)}
                    for item_negs in batch["neg"]:
                        item_negs_tokenized = [neg_dict[text] for text in item_negs]
                        negatives.append(item_negs_tokenized)
                else:
                    negatives.append([])
            else:
                for q in batch["pos"]:
                    negatives.append([])
            scores = []
            if "negscore" in batch and "neg" in batch:
                for neglist, negscorelist in zip(batch["neg"], batch["negscore"]):
                    if len(neglist) != len(negscorelist):
                        raise Exception(
                            f"number of neg/negscore mismatch for query  {len(neglist)} != {len(negscorelist)}"
                        )
                    else:
                        scores.append(negscorelist)
            elif "neg" in batch:
                for neglist in batch.get("neg"):
                    scores.append([0] * len(neglist))
            else:
                scores.append([])
            result = {
                "query": query,
                "query_text": batch["query"] if "query" in batch else [None] * len(batch["pos"]),
                "pos": positive,
                "pos_text": batch["pos"],
                "neg": negatives,
                "negscore": scores,
            }
            return result

        schema = Features(
            {
                "query": Sequence(Value("int32")),
                "query_text": Value("string"),
                "pos": Sequence(Value("int32")),
                "pos_text": Value("string"),
                "neg": [Sequence(Value("int32"))],
                "negscore": Sequence(Value("float")),
            }
        )
        return data.map(function=process_batch, batched=True, features=schema, num_proc=num_workers, desc="tokenizing")
