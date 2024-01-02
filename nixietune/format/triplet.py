from transformers import PreTrainedTokenizerBase
from typing import Dict, List, Optional
from nixietune.format import Format
from datasets import Features, Value


class TripletFormat(Format):
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

        docs = []
        lengths = []
        for q, p, n in zip(batch.get("query", []), batch.get("positive", []), batch.get("negative", [])):
            query_tokens = token_cache[self.format_query(q)]
            if len(n) > 0:
                for pos in p:
                    pos_tokens = token_cache[self.format_doc(pos)]
                    for neg in n[: self.neg_count]:
                        neg_tokens = token_cache[neg]
                        docs.append([query_tokens, pos_tokens, neg_tokens])
                        lengths.append(
                            pos_tokens["input_ids"].size
                            + query_tokens["input_ids"].size
                            + neg_tokens["input_ids"].size
                        )

        result = {"features": docs, "length": lengths}
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
                "length": Value("int32"),
            }
        )
