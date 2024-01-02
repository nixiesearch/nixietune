from mteb import MTEB
import torch
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
import sys
import os


class SBERTModel:
    def __init__(
        self,
        name: str,
        query_prefix: Optional[str] = None,
        doc_prefix: Optional[str] = None,
    ):
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.model = SentenceTransformer(name)
        self.model.to("cuda:0")
        self.model.eval()

    def encode_queries(self, queries, batch_size=32, **kwargs):
        if self.query_prefix is not None:
            processed = [f"{self.query_prefix}{q}" for q in queries]
        else:
            processed = queries
        return self.encode_impl(processed, batch_size)

    def encode_corpus(self, corpus, batch_size=32, **kwargs):
        if self.doc_prefix is not None:
            processed = [f"{self.query_prefix}{q}" for q in corpus]
        else:
            processed = corpus
        return self.encode_impl(processed, batch_size)

    @torch.no_grad()
    def encode_impl(self, input_texts, batch_size):
        return self.model.encode(input_texts, normalize_embeddings=True, batch_size=batch_size)


@dataclass
class EvalArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    result_path: str = field(metadata={"help": "Output path for results"})
    batch_size: int = field(default=256, metadata={"help": "Batch size for model inference"})


def main():
    parser = HfArgumentParser((EvalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (eval_args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (eval_args,) = parser.parse_args_into_dataclasses()

    model = SBERTModel(eval_args.model_name_or_path)
    evaluation = MTEB(tasks=["FiQA2018", "SciFact", "NFCorpus"])
    evaluation.run(model, output_folder=eval_args.result_path, batch_size=eval_args.batch_size, verbosity=2)


if __name__ == "__main__":
    main()
