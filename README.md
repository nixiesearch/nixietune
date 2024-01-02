# Nixietune: a fine-tuner for semantic search models

[![License: Apache 2](https://img.shields.io/badge/License-Apache2-green.svg)](https://opensource.org/licenses/Apache-2.0)
![Last commit](https://img.shields.io/github/last-commit/nixiesearch/nixietune)
![Last release](https://img.shields.io/github/release/nixiesearch/nixietune)
[![Join our slack](https://img.shields.io/badge/Slack-join%20the%20community-blue?logo=slack&style=social)](https://communityinviter.com/apps/nixiesearch/nixiesearch)

Nixietune is a GPU fine-tuning harness for semantic search models. Built for the [Nixiesearch search engine](https://github.com/nixiesearch/nixiesearch):

* a set of state-of-the-art recipes to fine-tune existing generic semantic search models like [E5](https://huggingface.co/intfloat/e5-base-v2)/[BGE](https://huggingface.co/BAAI/bge-base-en-v1.5)/[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) on your data
* based on battle-tested [sentence-transformers](https://github.com/UKPLab/sentence-transformers) library, but uses modern Huggingface ecosystem for training: multi-GPU and distributed training, FP16/BF16 mixed-precision, gradient checkpointing/accumulation and dataset caching.
* Can be used with and without hard negatives, supports InfoNCE/Cosine/Contrastive/Triples losses.

## Usage

To fine-tune a semantic search embedding model on your data:

* **Install nixietune**: you need a GPU for that!
* **Format your data in a nixietune format**: a JSONL file with a specific schema.
* **Run the training**: for base/small models it takes less than an hour on a single GPU.
* **Tinker with params**: choose the best loss and make your model training faster.

### Installation

Nixietune is not yet published to PyPi, but you can install it from git:

```bash
# get the code
git clone git@github.com:nixiesearch/nixietune.git
cd nixietune
# setup the environment
python -m venv .venv && source .venv/bin/activate
# install dependencies
pip install -r requirements.txt
```

* Nixietune is tested with Python 3.11. 
* 3.12 is not yet supported [by PyTorch](https://github.com/pytorch/pytorch/issues/110436)
* Python 3.10 and earlier: use at your own risk.

### Data format

Nixietune expects a specific JSONL input format for your documents:

```json
{
    "query": "pizza",
    "positive": [
        "Standard Serious Pizza",
        "60 Seconds to Napoli",
    ],
    "negative": [
        "Burgermeister",
        "Risa Chicken",
    ]
}
```

The document schema can be described as:

* `query`: required, string. An anchor search query for the whole group of documents.
* `pos`: required, list[string]. A one or more positive documents for the query above.
* `neg`: optional, list[string]. A zero or more negative documents for the query.

### Run the training

Let's fine-tune a [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embedding model on a [nixiesearch/ms-marco-hard-negatives](https://huggingface.co/datasets/nixiesearch/ms-marco-hard-negatives) dataset, using the InfoNCE loss. 

```shell
python examples/train_msmarco.py examples/msmarco.json
```

The [`msmarco.json`](examples/msmarco.json) configuration file is based on a HuggingFace Transformer TrainingArguments with some extra settings:

```json
{
    "seq_len": 128,
    "target": "infonce",
    "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
    "output_dir": "out",
    "num_train_epochs": 1,
    "seed": 33,
    "per_device_train_batch_size": 256,
    "per_device_eval_batch_size": 256,
    "fp16": true,
    "logging_dir": "logs",
    "gradient_checkpointing": true,
    "gradient_accumulation_steps": 1,
    "dataloader_num_workers": 14,
    "eval_steps": 0.1,
    "logging_steps": 0.1,
    "evaluation_strategy": "steps",
    "torch_compile": true,
    "report_to": [],
    "save_strategy": "epoch",
    "num_negatives": 8
}
```

It takes around 20 minutes to fine-tune an `all-MiniLM-L6-v2` on a MS MARCO hard negatives on a single RTX4090 GPU.

### Choosing the best parameters

The following training parameters are worth tuning:

* `target`: the training recipe. Currently supported targets are `infonce`/`cosine_similarity`/`contrastive`/`triplet`. If not sure, start with `infonce`.
* `model_name_or_path`: which model to fine-tune. Any SBERT-supported model should work.
* `per_device_train_batch_size`: batch size. Too small values lead to sub-par quality and slow training. Too large need a lot of VRAM. Start with 128 and go up.
* `seq_len`: context length of the model. Usually it's around 128-160 for most models in MTEB leaderboard.
* `gradient_checkpointing`: reduces VRAM usage sugnificantly (up to 70%) with a small 10% performance penalty, as we recompute gradients instead of storing them. If unsure, choose `true`
* `num_negatives`: for `infonce`/`triplet` targets, how many negatives from the dataset to select.

## License

Apache 2.0