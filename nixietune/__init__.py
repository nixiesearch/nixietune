from nixietune.biencoder import BiencoderTrainer, BiencoderTrainingArguments
from nixietune.format import Format, QueryDocLabelFormat, QueryPosNegsFormat, TripletFormat
from nixietune.arguments import ModelArguments, DatasetArguments
from datasets import DatasetDict, Dataset, load_dataset
import os
from huggingface_hub.utils import validate_repo_id
import logging

logger = logging.getLogger()


def load_dataset_split(path: str, split: str) -> Dataset:
    if os.path.exists(path):
        if os.path.isdir(path):
            return load_dataset("json", data_dir=path, split=split)
        else:
            return load_dataset("json", data_files={split: path}, split=split)
    else:
        validate_repo_id(path)
        return load_dataset(path, split=split)
