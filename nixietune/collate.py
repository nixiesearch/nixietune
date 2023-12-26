from typing import Any, List, Dict, Any
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding
import torch


class EmbeddingCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {"label": torch.Tensor([f["label"] for f in features])}
        for feature in ["sentence_A", "sentence_B"]:
            padded = self.pad(feature, features)
            batch[f"{feature}_input_ids"] = padded.input_ids
            batch[f"{feature}_attention_mask"] = padded.attention_mask
        return batch

    def pad(self, feature: str, features: List[Dict[str, Any]]) -> BatchEncoding:
        batch = BatchEncoding(
            data={
                "input_ids": [f[f"{feature}.input_ids"] for f in features],
                "attention_mask": [f[f"{feature}.attention_mask"] for f in features],
            }
        )
        return self.tokenizer.pad(
            batch, padding="longest", pad_to_multiple_of=8, return_tensors="pt"
        )
