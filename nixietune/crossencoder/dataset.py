from transformers import PreTrainedTokenizerBase, BatchEncoding
from datasets import Dataset
from typing import Dict, List, Any, Optional
import random
import torch
from collections import defaultdict


class CrossEncoderFormat:
    def collate(self, items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pass

    def prepare_dataset(
        self,
        dataset: Dataset,
        num_negatives: int,
        num_workers: int = 1,
        neg_strategy: str = "random",
    ) -> Dataset:
        pass


class QueryDocListFormat(CrossEncoderFormat):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def prepare_dataset(
        self,
        dataset: Dataset,
        num_negatives: int,
        num_workers: int = 1,
        neg_strategy: str = "random",
    ) -> Dataset:
        def ranker_layout(batch: Dict[str, List]) -> Dict[str, List]:
            query_groups = []
            labels = []
            for query, pos, negs, negscores in zip(batch["query"], batch["doc"], batch["neg"], batch["negscore"]):
                if len(negs) > 0:
                    match neg_strategy:
                        case "random":
                            neg_sample = random.choices(negs, k=num_negatives)
                        case "first":
                            neg_sample = negs[:num_negatives]
                        case other:
                            raise Exception(f"sampling method {other} not supported")
                    if len(neg_sample) == num_negatives:
                        positive_position = random.randint(0, num_negatives)
                        neg_sample.insert(positive_position, pos)
                        prompt = "<s>".join([query] + neg_sample)
                        query_groups.append(prompt)
                        labels.append(positive_position)
            result = self.tokenizer(query_groups, padding=False, truncation=True, max_length=self.max_len)
            result["labels"] = labels
            return result

        return dataset.map(
            function=ranker_layout,
            batched=True,
            desc="tokenizing",
            remove_columns=["query", "doc", "neg", "negscore"],
            num_proc=num_workers,
            batch_size=4,
        )

    def collate(self, items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        data = defaultdict(list)
        for item in items:
            for key, value in item.items():
                data[key].append(value)
        batch = BatchEncoding(data=data)
        encoded = self.tokenizer.pad(batch, padding="longest", pad_to_multiple_of=8, return_tensors="pt")
        encoded["labels"] = torch.tensor([input["labels"] for input in items])
        return encoded


class QueryDocTupleFormat(CrossEncoderFormat):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def prepare_dataset(
        self,
        dataset: Dataset,
        num_negatives: Optional[int],
        num_workers: int = 1,
        neg_strategy: str = "random",
    ) -> Dataset:
        def cross_layout(batch: Dict[str, List]) -> Dict[str, List]:
            pairs = []
            labels = []
            for query, pos, negs, negscores in zip(batch["query"], batch["doc"], batch["neg"], batch["negscore"]):
                pairs.append((query, pos))
                labels.append(1.0)
                if len(negs) > 0:
                    if num_negatives:
                        match neg_strategy:
                            case "random":
                                neg_sample = random.choices(list(zip(negs, negscores)), k=num_negatives)
                            case "first":
                                neg_sample = zip(negs, negscores)[:num_negatives]
                            case other:
                                raise Exception(f"sampling method {other} not supported")
                        for neg, score in neg_sample:
                            pairs.append((query, neg))
                            labels.append(score)
                    else:
                        for neg, score in zip(negs, negscores):
                            pairs.append((query, neg))
                            labels.append(score)

            result = self.tokenizer(pairs, padding=False, truncation=True, max_length=self.max_len)
            result["labels"] = labels
            return result

        return dataset.map(
            function=cross_layout,
            batched=True,
            desc="tokenizing",
            remove_columns=["query", "doc", "neg", "negscore"],
            num_proc=num_workers,
        )

    def collate(self, items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        data = defaultdict(list)
        for item in items:
            for key, value in item.items():
                data[key] = [value]
        batch = BatchEncoding(data=data)
        encoded = self.tokenizer.pad(batch, padding="longest", pad_to_multiple_of=8, return_tensors="pt")
        encoded["labels"] = torch.tensor([input["labels"] for input in items])
        return encoded
