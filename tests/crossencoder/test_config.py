from transformers import HfArgumentParser
from nixietune.arguments import ModelArguments, DatasetArguments
from nixietune.crossencoder.arguments import CrossEncoderArguments


def test_conf_nested():
    parser = HfArgumentParser((ModelArguments, DatasetArguments, CrossEncoderArguments))
    conf = "tests/crossencoder/config/ce_lora.json"
    model_args, dataset_args, training_args = parser.parse_json_file(conf)
