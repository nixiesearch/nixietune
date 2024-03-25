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
    lora: bool = field(default=False, metadata={"help": "enable LoRA"})
    lora_r: int = field(default=16, metadata={"help": "R value for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha value for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout value for LoRA"})
    lora_load_bits: int = field(default=4, metadata={"help": "model weights precision for QLoRA"})
    attn_implementation: str = field(
        default="eager", metadata={"help": "Which attention impl to use. Try flash_attention_2"}
    )
