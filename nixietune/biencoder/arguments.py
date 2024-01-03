from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import Optional


@dataclass
class BiencoderTrainingArguments(TrainingArguments):
    seq_len: int = field(
        default=128,
        metadata={"help": "Max sequence length in tokens."},
    )

    target: str = field(
        default="infonce", metadata={"help": "Optimization target: cosine_similarity/contrastive/infonce"}
    )

    num_negatives: int = field(default=4, metadata={"help": "Number of negatives to use for InfoNCE/Triplet loss"})

    triplet_margin: float = field(default=5, metadata={"help": "Margin value for Triplet loss"})

    query_prefix: Optional[None] = field(
        default=None, metadata={"help": "Prefix for all queries. Used for asymmetrical models like E5."}
    )

    document_prefix: Optional[None] = field(
        default=None, metadata={"help": "Prefix for all documents. Used for asymmetrical models like E5."}
    )

    infonce_negative_mode: str = field(
        default="unpaired",
        metadata={
            "help": "Negative loss mode: paired (when loss is computed per query) / unpaired (when all negatives for all queries are taken at once)"
        },
    )

    infonce_temperature: float = field(default=0.05, metadata={"help": "Temperature for InfoNCE loss"})
