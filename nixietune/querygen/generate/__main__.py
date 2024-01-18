from nixietune.querygen.generate.generate import QueryGenerator, GeneratorArguments, DatasetArguments
from nixietune.log import setup_logging
import logging
from transformers import HfArgumentParser

setup_logging()


if __name__ == "__main__":
    logger = logging.getLogger()
    parser = HfArgumentParser((GeneratorArguments, DatasetArguments))
    gen_args, data_args = parser.parse_args_into_dataclasses()

    gen = QueryGenerator(gen_args)
    gen.generate(data_args.input_path).to_json(data_args.output_path)
