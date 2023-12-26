from sentence_transformers import SentenceTransformer, losses, util
from nixietune.format import TripletDataset
from transformers import AutoTokenizer, TrainingArguments
from torch.utils.data import DataLoader
from nixietune.collate import EmbeddingCollator
from datasets import load_dataset, Dataset
from nixietune.model.biencoder import BiencoderTrainer, BiencoderModel
from typing import Dict
from transformers.trainer_utils import EvalPrediction
import logging
from torchmetrics.retrieval import RetrievalNormalizedDCG
import torch

log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)
logger.handlers = [console_handler]


path = "/home/shutty/data/esci"
files = {"train": f"{path}/train.jsonl", "test": f"{path}/test5.jsonl"}

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = model.tokenizer
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

train = load_dataset("json", data_files=files, split="train", num_proc=8)
test = load_dataset("json", data_files=files, split="test", num_proc=8)
data_train = TripletDataset(dataset=train, tokenizer=tokenizer, seq_len=128)
data_test = TripletDataset(dataset=test, tokenizer=tokenizer, seq_len=128)
collator = EmbeddingCollator(tokenizer)


def tocpu(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    result = []
    for f in logits:
        se = f["sentence_embedding"]
        result.append({"sentence_embedding": se.cpu()})
    return result


def compute_metrics(embeds: EvalPrediction) -> Dict[str, float]:
    query_embeds = torch.from_numpy(embeds.predictions[0]["sentence_embedding"]).cpu()
    doc_embeds = torch.from_numpy(embeds.predictions[1]["sentence_embedding"]).cpu()
    scores = util.pairwise_cos_sim(query_embeds, doc_embeds)
    _, indices = torch.unique_consecutive(query_embeds, return_inverse=True, dim=0)
    ndcg = RetrievalNormalizedDCG(top_k=10)
    result = ndcg(scores, torch.from_numpy(embeds.label_ids), indexes=indices)
    return {"ndcg": result.item()}


# loss = losses.CosineSimilarityLoss(model)
loss = losses.ContrastiveLoss(model)
training_arguments = TrainingArguments(
    report_to=None,
    output_dir=".",
    num_train_epochs=1,
    seed=33,
    per_device_train_batch_size=512,
    per_device_eval_batch_size=512,
    fp16=True,
    # checkpoint settings
    logging_dir="logs",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # needed to get sentence_A and sentence_B
    remove_unused_columns=False,
    dataloader_num_workers=8,
    eval_steps=500,
    logging_steps=500,
    evaluation_strategy="steps",
    label_names=["label"],
    torch_compile=True,
    include_tokens_per_second=False,
    include_num_input_tokens_seen=False,
    save_strategy="no",
)

trainer = BiencoderTrainer(
    model=BiencoderModel(model),
    args=training_arguments,
    train_dataset=data_train.dataset,
    eval_dataset=data_test.dataset,
    data_collator=collator,
    tokenizer=tokenizer,
    loss=loss,
    text_columns=["sentence_A", "sentence_B"],
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=tocpu,
)
trainer.train(ignore_keys_for_eval=["input_ids", "attention_mask"])
