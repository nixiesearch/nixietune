from datasets import Features, Value
from transformers import PreTrainedTokenizerBase
from typing import Dict, List, Optional
from nixietune.format import Format


class QueryDocLabelFormat(Format):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        query_prefix: Optional[str] = None,
        doc_prefix: Optional[str] = None,
        neg_count: int = 8,
    ) -> None:
        super().__init__(tokenizer, query_prefix, doc_prefix)
        self.neg_count = neg_count

    def tokenize(self, batch: Dict[str, List]) -> Dict[str, List]:
        token_cache = self.make_tokenized_cache(batch)

        features = []
        lengths = []
        labels = []
        for q, p, n in zip(batch.get("query", []), batch.get("positive", []), batch.get("negative", [])):
            query_tokens = token_cache[self.format_query(q)]
            for pos in p:
                doc = self.format_doc(pos)
                doc_tokens = token_cache[doc]
                features.append([query_tokens, doc_tokens])
                lengths.append(query_tokens["input_ids"].size + doc_tokens["input_ids"].size)
                labels.append(1.0)
            for neg in n:
                doc = self.format_doc(neg)
                doc_tokens = token_cache[doc]
                features.append([query_tokens, doc_tokens])
                lengths.append(query_tokens["input_ids"].size + doc_tokens["input_ids"].size)
                labels.append(0.0)
        result = {"features": features, "label": labels, "length": lengths}
        return result

    def schema(self, dtype: str) -> Features:
        return Features(
            {
                "features": [
                    {
                        "input_ids": [Value(dtype)],
                        "attention_mask": [Value("int8")],
                    }
                ],
                "label": Value("float"),
                "length": Value("int32"),
            }
        )
