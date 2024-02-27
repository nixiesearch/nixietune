from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetArguments:
    train_dataset: str = field(metadata={"help": "Path to training dataset in TREC format"})
    train_split: str = field(default="train", metadata={"help": "name of the qrel split for training"})
    eval_split: str = field(default="test", metadata={"help": "name of the qrel split for evaluation"})
    eval_dataset: Optional[str] = field(default=None, metadata={"help": "Path to evaluation dataset in TREC format"})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
