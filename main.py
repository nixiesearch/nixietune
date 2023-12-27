from sentence_transformers import SentenceTransformer, losses, util
from nixietune.trainer import BiencoderTrainer, BiencoderTrainingArguments
from nixietune.log import setup_logging
from datasets import load_dataset

setup_logging()

path = "/home/shutty/data/esci"
files = {"train": f"{path}/train.jsonl", "test": f"{path}/test5.jsonl"}
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dataset = load_dataset("json", data_files=files, num_proc=8)

training_arguments = BiencoderTrainingArguments(
    seq_len=128,
    target="contrastive",
    report_to=None,
    output_dir=".",
    num_train_epochs=1,
    seed=33,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    fp16=True,
    # checkpoint settings
    logging_dir="logs",
    gradient_checkpointing=False,
    # gradient_checkpointing_kwargs={"use_reentrant": False},
    # needed to get sentence_A and sentence_B
    remove_unused_columns=False,
    dataloader_num_workers=14,
    eval_steps=100,
    logging_steps=100,
    evaluation_strategy="steps",
    label_names=["label"],
    torch_compile=True,
    save_strategy="no",
)

trainer = BiencoderTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)
trainer.train()
