from transformers import LlamaTokenizer
from typing import Dict, List
import numpy as np


class DatasetTokenizer:
    def __init__(self, tokenizer: LlamaTokenizer, seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.question_kws = [
            "which",
            "on which",
            "on when",
            "what",
            "who",
            "how",
            "why",
            "where",
            "are",
            "is",
            "when",
            "if",
            "can",
            "do",
            "does",
            "in what",
            "has",
            "have",
            "had",
        ]

    def tokenize(self, batch: Dict[str, List[str]]) -> Dict[str, List]:
        queries = []
        for query in batch["query_text"]:
            # length_prefix = self.length_type(query)
            # type_prefix = self.question_type(query)
            # processed_query = f"{length_prefix} {type_prefix} query: {query}"
            processed_query = f"query: {query}"
            queries.append(processed_query)
        tokenized_queries = self.tokenizer(
            queries,
            padding=False,
            truncation=True,
            max_length=self.seq_len,
            return_overflowing_tokens=False,
            return_length=False,
        )

        passages = batch["pos_text"]
        tokenized_passages = self.tokenizer(
            passages,
            padding=False,
            truncation=True,
            max_length=self.seq_len,
            return_overflowing_tokens=False,
            return_length=False,
        )

        inputs = []
        attmasks = []
        for q_input, doc_input in zip(tokenized_queries["input_ids"], tokenized_passages["input_ids"]):
            max_doc_len = self.seq_len - len(q_input) - 2
            input = [self.tokenizer.bos_token_id] + doc_input[:max_doc_len] + q_input + [self.tokenizer.eos_token_id]
            attmask = [1] * len(input)
            inputs.append(input)
            attmasks.append(attmask)
        return {"input_ids": inputs, "attention_mask": attmasks}

    def length_type(self, query: str) -> str:
        if np.random.randint(100) < 50:
            words = len(query.split())
            match words:
                case x if x < 5:
                    return "short"
                case x if x < 10:
                    return "medium"
                case _:
                    return "long"
        else:
            return ""

    def question_type(self, query: str) -> str:
        if np.random.randint(100) < 50:
            lowercase_query = query.lower().strip()
            if any(lowercase_query.startswith(prefix) for prefix in self.question_kws):
                return "question"
            elif lowercase_query.endswith("?"):
                return "question"
            else:
                return "regular"
        else:
            return ""
