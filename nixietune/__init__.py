from nixietune.biencoder import BiencoderTrainer, BiencoderTrainingArguments  # noqa: F401
from nixietune.format import Format, QueryDocLabelFormat, QueryPosNegsFormat, TripletFormat  # noqa: F401
from nixietune.arguments import ModelArguments, DatasetArguments  # noqa: F401
from datasets import Dataset, load_dataset, Features, Value
import os
from huggingface_hub.utils import validate_repo_id
import logging
from typing import Optional

logger = logging.getLogger()


def load_dataset_split(path: str, split: str, samples: Optional[int] = None, streaming: bool = False) -> Dataset:
    schema = Features({"query": Value("string"), "positive": [Value("string")], "negative": [Value("string")]})
    if os.path.exists(path):
        if os.path.isdir(path):
            dataset = load_dataset("json", data_dir=path, split=split, features=schema, streaming=streaming)
        else:
            dataset = load_dataset("json", data_files={split: path}, split=split, features=schema, streaming=streaming)
    else:
        validate_repo_id(path)
        dataset = load_dataset(path, split=split, features=schema, streaming=streaming)
    if samples is not None and streaming is False:
        dataset = dataset.select(list(range(samples)))
    return dataset
