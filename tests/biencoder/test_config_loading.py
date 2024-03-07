from nixietune import ModelArguments, DatasetArguments, BiencoderTrainingArguments
from transformers import HfArgumentParser


def test_load_esci():
    parser = HfArgumentParser((ModelArguments, DatasetArguments, BiencoderTrainingArguments))
    model_args, dataset_args, training_args = parser.parse_json_file(
        json_file="tests/biencoder/config/esci_infonce.json"
    )
    assert model_args.model_name_or_path == "sentence-transformers/all-MiniLM-L6-v2"
