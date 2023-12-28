# Nixietune: a fine-tuner for semantic search models

[![License: Apache 2](https://img.shields.io/badge/License-Apache2-green.svg)](https://opensource.org/licenses/Apache-2.0)
![Last commit](https://img.shields.io/github/last-commit/nixiesearch/nixietune)
![Last release](https://img.shields.io/github/release/nixiesearch/nixietune)
[![Join our slack](https://img.shields.io/badge/Slack-join%20the%20community-blue?logo=slack&style=social)](https://communityinviter.com/apps/nixiesearch/nixiesearch)

Nixietune is a GPU fine-tuning harness for semantic search models:

* a set of state-of-the-art recipes to fine-tune existing generic semantic search models like [E5](https://huggingface.co/intfloat/e5-base-v2)/[BGE](https://huggingface.co/BAAI/bge-base-en-v1.5)/[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) on your data
* based on battle-tested [sentence-transformers](https://github.com/UKPLab/sentence-transformers) library, but uses modern Huggingface ecosystem for training: multi-GPU and distributed training, FP16/BF16 mixed-precision, gradient checkpointing/accumulation and dataset caching.
* Can be used with and without hard negatives, supports InfoNCE/Cosine/Contrastive losses.

## Usage

To fine-tune a semantic search embedding model on your data, you need:

* Install nixietune: you need a GPU for that!
* Format your data in a nixietune format: a JSONL file with a specific schema.
* Run the training: for base/small models it takes less than an hour on a single GPU.
* Tinker with params: choose the best loss and make your model training faster.

### Installation

Nixietune is not yet published to PyPi, but you can install it from git:

```bash
git clone git@github.com:nixiesearch/nixietune.git
cd nixietune
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nixietune is tested with Python 3.11. 3.12 is not yet supported [by PyTorch](https://github.com/pytorch/pytorch/issues/110436)

### Data format

Nixietune expects a specific JSONL input format for your documents:

```json
{
    "query": "pizza",
    "pos": [
        {"doc": "Standard Serious Pizza", "score": 1.0},
        {"doc": "60 Seconds to Napoli", "score": 1.0},
    ],
    "neg": [
        {"doc": "Burgermeister", "score": 0.0},
        {"doc": "Risa Chicken", "score": 0.0}
    ]
}
```

The document schema can be described as:

* `query`: required, string. An anchor search query for the whole group of documents.
* `pos`: required, list[obj]. A one or more positive documents for the query above.
* `pos.doc`: required, string. A document text.
* `pos.score`: optional, float. On optional relevancy score for the document. If not present, all positive documents have an implicit score of `1.0`.
* `neg`: optional, list[obj]. A zero or more negative documents for the query.
* `neg.doc`: required, string. A document text.
* `neg.score`: optional, float. On optional relevancy score for the document. If not present, all negative documents have an implicit score of `0.0`.

### Run the training

todo

### Choosing the best parameters

todo

## License

Apache 2.0