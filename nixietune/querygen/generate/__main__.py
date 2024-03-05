from nixietune.querygen.generate.generate import QueryGenerator, GeneratorArguments, DatasetArguments
from nixietune.log import setup_logging
import logging
from transformers import HfArgumentParser
import json

setup_logging()


if __name__ == "__main__":
    logger = logging.getLogger()
    parser = HfArgumentParser((GeneratorArguments, DatasetArguments))
    gen_args, data_args = parser.parse_args_into_dataclasses()

    gen = QueryGenerator(gen_args)

    with open(data_args.output_file, "w") as f:
        for doc in gen.generate(data_args.input_file):
            f.write(json.dumps(doc) + "\n")
    # TRECDatasetWriter.save(out_path=data_args.output_path, dataset=result)
