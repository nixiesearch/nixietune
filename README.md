# Nixietune: a fine-tuner for semantic search models

[![License: Apache 2](https://img.shields.io/badge/License-Apache2-green.svg)](https://opensource.org/licenses/Apache-2.0)
![Last commit](https://img.shields.io/github/last-commit/nixiesearch/nixietune)
![Last release](https://img.shields.io/github/release/nixiesearch/nixietune)
[![Join our slack](https://img.shields.io/badge/Slack-join%20the%20community-blue?logo=slack&style=social)](https://communityinviter.com/apps/nixiesearch/nixiesearch)

Nixietune is a GPU fine-tuning harness for semantic search models. Built for the [Nixiesearch search engine](https://github.com/nixiesearch/nixiesearch):

* a set of state-of-the-art recipes to fine-tune existing generic semantic search models like [E5](https://huggingface.co/intfloat/e5-base-v2)/[BGE](https://huggingface.co/BAAI/bge-base-en-v1.5)/[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) on your data
* based on battle-tested [sentence-transformers](https://github.com/UKPLab/sentence-transformers) library, but uses modern Huggingface ecosystem for training: multi-GPU and distributed training, FP16/BF16 mixed-precision, gradient checkpointing/accumulation and dataset caching.
* Can be used with and without hard negatives, supports InfoNCE/Cosine/Contrastive/Triples losses.

## Features

What Nixietune can do for you:

* Fine-tune an existing embedding model on your labeled data.
* Generate synthetic queries and labels
* Train a cross-encoder reranker model.

## Usage

### Fine-tuning an embedding model

To fine-tune a semantic search embedding model on your data:

* **Install nixietune**: you need a GPU for that!
* **Format your data in a nixietune format**: a JSON file format with a specific schema.
* **Run the training**: for base/small models it takes less than an hour on a single desktop GPU.
* **Tinker with params**: choose the best loss and make your model training faster.

### Installation

Nixietune is published to PyPi:

```bash
# setup the environment
python -m venv .venv && source .venv/bin/activate
# install dependencies
pip install nixietune
```

* Nixietune is tested with Python 3.10 and 3.11. 
* 3.12 is not yet supported [by PyTorch](https://github.com/pytorch/pytorch/issues/110436)

### Data format

Nixietune expects a specific JSONL input format for your documents:

```json
{
    "query": "pizza",
    "doc": "Standard Serious Pizza",
    "neg": [
        "Burgermeister",
        "Risa Chicken",
    ]
}
```

The document schema can be described as:

* `query`: `string`. An anchor search query for the whole group of documents.
* `doc`: `string`. A one or more positive documents for the query above.
* `neg`: `list[string]`. A zero or more negative documents for the query.
* `negscore`: `list[float]`. A zero or more scores for negatives.

All fields are formally optional and different modules require different fields, but for a traditional embedding fine-tuning we need `query`, `doc` and optionally `neg` fields to be present.

Some losses like InfoNCE can be trained without negatives (so you need only `query` and `doc` fields in the training data), but usually you can get much better results with explicit negatives.

### Run the training

Let's fine-tune a [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embedding model on a [nixiesearch/amazon-esci](https://huggingface.co/datasets/nixiesearch/amazon-esci) dataset, using the InfoNCE loss. 

```shell
python -m nixietune.biencoder examples/esci.json
```

The [`esci.json`](examples/esci.json) configuration file is based on a HuggingFace Transformer TrainingArguments with some extra settings:

```json
{
    "seq_len": 128,
    "target": "infonce",
    "train_dataset": "nixiesearch/amazon-esci",
    "eval_dataset": "nixiesearch/amazon-esci",
    "train_split": "train[:10%]",
    "eval_split": "test_1k",
    "model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
    "output_dir": "out",
    "num_train_epochs": 1,
    "seed": 33,
    "per_device_train_batch_size": 512,
    "per_device_eval_batch_size": 512,
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
    "num_negatives": 8,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "learning_rate": 5e-5
```

It takes around 60 minutes to fine-tune an `all-MiniLM-L6-v2` on an Amazon ESCI dataset on a single RTX4090 GPU.

### Choosing the best parameters

The following training parameters are worth tuning:

* `target`: the training recipe. Currently supported targets are `infonce`/`cosine_similarity`/`contrastive`/`triplet`. If not sure, start with `infonce`.
* `model_name_or_path`: which model to fine-tune. Any SBERT-supported model should work.
* `per_device_train_batch_size`: batch size. Too small values lead to sub-par quality and slow training. Too large need a lot of VRAM. Start with 128 and go up.
* `seq_len`: context length of the model. Usually it's around 128-160 for most models in MTEB leaderboard.
* `gradient_checkpointing`: reduces VRAM usage sugnificantly (up to 70%) with a small 10% performance penalty, as we recompute gradients instead of storing them. If unsure, choose `true`
* `num_negatives`: for `infonce`/`triplet` targets, how many negatives from the dataset to select.
* `query_prefix` and `document_prefix`: prompt labels for asymmetric models like E5 - when the model can distinguish between query and document passages.

## Training a cross-encoder

Cross-encoders are not limited by the restrictions of cosine space, and usually provide much more precise result - for the extra cost of much resource-hungry inference. 

Training a cross-encoder with `nixietune` requires negatives to be present in your data (so `query`, `doc` and `neg` fields) and is possible with the following config file:

```json
{
    "seq_len": 128,
    "train_dataset": "nixiesearch/amazon-esci",
    "eval_dataset": "nixiesearch/amazon-esci",
    "train_split": "train",
    "eval_split": "test_1k",
    "model_name_or_path": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "output_dir": "out",
    "num_train_epochs": 1,
    "seed": 33,
    "per_device_train_batch_size": 1024,
    "per_device_eval_batch_size": 1024,
    "fp16": true,
    "logging_dir": "logs",
    "gradient_checkpointing": true,
    "gradient_accumulation_steps": 1,
    "dataloader_num_workers": 14,
    "eval_steps": 0.1,
    "logging_steps": 0.1,
    "evaluation_strategy": "steps",
    "torch_compile": false,
    "report_to": [],
    "save_strategy": "epoch",
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "learning_rate": 5e-5
}
```

It can be launched with the following command:

```shell
python -m nixietune.crossencoder examples/esci_ce.json
```

## License

Apache 2.0