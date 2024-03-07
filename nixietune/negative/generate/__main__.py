from nixietune.log import setup_logging
from transformers import HfArgumentParser
from nixietune.negative.generate.args import GenArgs
from nixietune.negative.generate.gen import NegativeGenerator

setup_logging()


if __name__ == "__main__":
    parser = HfArgumentParser((GenArgs))
    (args,) = parser.parse_args_into_dataclasses()
    generator = NegativeGenerator(args)
