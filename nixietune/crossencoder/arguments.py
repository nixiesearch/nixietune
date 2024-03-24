from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List, Optional


@dataclass
class LoraArguments:
    r: int = field(default=16, metadata={"help": "R value for LoRA"})
    alpha: int = field(default=32, metadata={"help": "alpha value for LoRA"})
    dropout: float = field(default=0.05, metadata={"help": "dropout value for LoRA"})


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
    lora: Optional[LoraArguments] = field(default=None, metadata={"help": "LoRA parameters"})
