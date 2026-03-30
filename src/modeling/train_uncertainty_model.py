"""
Fine-tune ProsusAI/finbert for binary uncertainty classification.

Reads: data/processed/train.parquet, val.parquet
Outputs: models/uncertainty_finbert/ (saved model + tokenizer)
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01


class SentenceDataset(Dataset):
    """PyTorch dataset for sentence classification."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def compute_metrics(eval_pred):
    """Compute classification metrics for the Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary"),
        "precision": precision_score(labels, predictions, average="binary"),
        "recall": recall_score(labels, predictions, average="binary"),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    data_dir = os.path.join(project_root, "data", "processed")
    model_dir = os.path.join(project_root, "models", "uncertainty_finbert")
    os.makedirs(model_dir, exist_ok=True)

    train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
    log.info("Train: %d, Val: %d", len(train_df), len(val_df))

    train_texts = train_df["sentence"].tolist()
    train_labels = train_df["binary_label"].tolist()
    val_texts = val_df["sentence"].tolist()
    val_labels = val_df["binary_label"].tolist()

    log.info("Loading %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    train_dataset = SentenceDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentenceDataset(val_texts, val_labels, tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Training on: %s", device)

    # Training arguments
    n_train_steps = (len(train_dataset) // BATCH_SIZE + 1) * NUM_EPOCHS
    warmup_steps = int(n_train_steps * WARMUP_RATIO)
    training_args = TrainingArguments(
        output_dir=os.path.join(model_dir, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    log.info("Starting training...")
    trainer.train()

    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    log.info("Model saved to %s", model_dir)

    # Final validation metrics
    metrics = trainer.evaluate()
    log.info("Final validation metrics: %s", metrics)


if __name__ == "__main__":
    main()
