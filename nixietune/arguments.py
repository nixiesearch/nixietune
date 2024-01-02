from dataclasses import dataclass, field


@dataclass
class DatasetArguments:
    train_dataset: str = field(metadata={"help": "Path to training dataset"})
    eval_dataset: str = field(metadata={"help": "Path to evaluation dataset"})
    train_split: str = field(default="train", metadata={"help": "name of the split for training"})
    eval_split: str = field(default="test", metadata={"help": "name of the split for evaluation"})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
