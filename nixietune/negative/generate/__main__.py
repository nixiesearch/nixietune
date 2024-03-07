import sys
from nixietune.log import setup_logging
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import hnswlib

setup_logging()


@dataclass
class GenArgs:
    num_negatives: int = field(default=32, metadata={"help": "number of negs"})
    model_name: str = field(metadata={"help": "embedding model"})
    query_file: str = field(metadata={"help": "input file with queries"})
    doc_file: str = field(metadata={"help": "input file with docs"})
    query_prefix: str = field(default="", metadata={"help": "query prefix"})
    doc_prefix: str = field(default="", metadata={"help": "doc prefix"})


if __name__ == "__main__":
    parser = HfArgumentParser((GenArgs))
    (args,) = parser.parse_args_into_dataclasses()
