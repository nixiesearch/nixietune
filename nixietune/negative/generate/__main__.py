from nixietune.log import setup_logging
from transformers import HfArgumentParser
from nixietune.negative.generate.gen import NegativeGenerator
from dataclasses import dataclass, field
import logging
import json
from tqdm import tqdm


@dataclass
class GenArgs:
    model_name: str = field(metadata={"help": "embedding model"})
    query_file: str = field(metadata={"help": "input file with queries"})
    doc_file: str = field(metadata={"help": "input file with docs"})
    out_file: str = field(metadata={"help": "path to output file"})
    query_prefix: str = field(default="", metadata={"help": "query prefix"})
    doc_prefix: str = field(default="", metadata={"help": "doc prefix"})
    num_negatives: int = field(default=16, metadata={"help": "number of negs"})
    batch_size: int = field(default=128, metadata={"help": "inference batch size"})


setup_logging()
logger = logging.getLogger()

if __name__ == "__main__":
    parser = HfArgumentParser((GenArgs))
    (args,) = parser.parse_args_into_dataclasses()
    generator = NegativeGenerator(query_file=args.query_file, doc_file=args.doc_file, model_name=args.model_name)
    results = generator.generate(args.num_negatives, args.batch_size)
    with open(args.out_file, "w") as f:
        for row in tqdm(results, desc="saving negatives"):
            line = json.dumps({"query": row.query, "doc": row.doc, "neg": row.neg})
            f.write(line + "\n")
    logger.info("generation done")
