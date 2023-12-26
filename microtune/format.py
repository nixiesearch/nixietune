from transformers import PreTrainedTokenizerBase
from datasets import load_dataset, DatasetDict, Value
from datasets import Dataset
from typing import Union, Optional, Dict, List, Any
from transformers.utils import PaddingStrategy
from dataclasses import dataclass
from datasets.features import Features


class TripletDataset:
    def __init__(
        self, dataset: Dataset, tokenizer: PreTrainedTokenizerBase, seq_len: int
    ) -> None:
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = seq_len
        self.dataset = dataset
        self.dataset = self.dataset.map(
            self.process_batch,
            remove_columns=["query", "pos", "neg"],
            # load_from_cache_file=False,
            batched=True,
            num_proc=8,
        )
        print(self.dataset.features)

    def process_batch(self, batch: Dict[str, List]) -> Dict[str, List]:
        queries: List[str] = []
        docs: List[str] = []
        labels: List[float] = []
        for q, p, n in zip(batch["query"], batch["pos"], batch["neg"]):
            for doc in p:
                queries.append(q)
                docs.append(doc["doc"])
                labels.append(doc.get("score", 1))
            for doc in n:
                queries.append(q)
                docs.append(doc["doc"])
                labels.append(doc.get("score", 0))
        qtokens = self.tokenizer(queries, padding=False, truncation=True)
        dtokens = self.tokenizer(docs, padding=False, truncation=True)
        return {
            "sentence_A.input_ids": qtokens.input_ids,
            "sentence_A.attention_mask": qtokens.attention_mask,
            "sentence_B.input_ids": dtokens.input_ids,
            "sentence_B.attention_mask": dtokens.attention_mask,
            "label": labels,
        }
