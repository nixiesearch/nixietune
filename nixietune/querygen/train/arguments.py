from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class QueryGenArguments(TrainingArguments):
    seq_len: int = field(
        default=128,
        metadata={"help": "Max sequence length in tokens."},
    )
