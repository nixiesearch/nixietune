from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetArguments:
    train_dataset: str = field(metadata={"help": "Path to training dataset"})
    train_split: str = field(default="train", metadata={"help": "name of the split for training"})
    eval_split: str = field(default="test", metadata={"help": "name of the split for evaluation"})
    eval_dataset: Optional[str] = field(default=None, metadata={"help": "Path to evaluation dataset"})
    train_samples: Optional[int] = field(
        default=None, metadata={"help": "Number of rows to select from the train split"}
    )
    eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Number of rows to select from the eval split"}
    )
    streaming: bool = field(default=False, metadata={"help": "Load dataset in streaming mode"})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
