from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List


@dataclass
class CrossEncoderArguments(TrainingArguments):
    seq_len: int = field(
        default=128,
        metadata={"help": "Max sequence length in tokens."},
    )

    eval_metrics: List[str] = field(
        default_factory=lambda: ["ndcg@10"], metadata={"help": "metrics to eval during training"}
    )

    num_negatives: int = field(default=4, metadata={"help": "number of negatives to sample"})
