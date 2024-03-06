import sys
from nixietune.log import setup_logging
from dataclasses import dataclass, field

setup_logging()


@dataclass
class GenArgs:
    num_negatives: int = field(default=32, metadata={"help": "number of negs"})
    model_name: str = field(metadata={"help": "embedding model"})
    input_file: str = field(metadata={"help": "input file with query+pos"})
    query_prefix: str = field(default="", metadata={"help": "query prefix"})
    doc_prefix: str = field(default="", metadata={"help": "doc prefix"})


if __name__ == "__main__":
    main(sys.argv)
