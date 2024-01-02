## Training examples

To run any of these example configurations, use the following command:

```bash
python -m nixietune examples/msmarco.json
```

### msmarco.json

A sample training config for fine-tuning the [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) with the following options:
* use the hard-negatives from [nixiesearch/ms-marco-hard-negatives](https://huggingface.co/datasets/nixiesearch/ms-marco-hard-negatives) as training dataset
* InfoNCE loss with 8 negatives per query

