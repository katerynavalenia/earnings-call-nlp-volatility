"""
Evaluate the fine-tuned uncertainty classification model on the test set.

Reads: data/processed/test.parquet, models/uncertainty_finbert/
Outputs: output/tables/model_evaluation.csv, printed metrics + examples
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BATCH_SIZE = 64
MAX_LENGTH = 128


def predict_batch(texts, tokenizer, model, device):
    """Get predictions and probabilities for a batch of texts."""
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=-1)

    return probs.cpu().numpy()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    model_dir = os.path.join(project_root, "models", "uncertainty_finbert")
    test_path = os.path.join(project_root, "data", "processed", "test.parquet")
    output_dir = os.path.join(project_root, "output", "tables")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_dir):
        log.error("Model not found at %s. Run train_uncertainty_model.py first.", model_dir)
        sys.exit(1)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading model from %s (device: %s)", model_dir, device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Load test data
    test_df = pd.read_parquet(test_path)
    log.info("Test set: %d sentences", len(test_df))

    texts = test_df["sentence"].tolist()
    true_labels = test_df["binary_label"].values

    # Predict in batches
    all_probs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Predicting"):
        batch_texts = texts[i : i + BATCH_SIZE]
        probs = predict_batch(batch_texts, tokenizer, model, device)
        all_probs.append(probs)

    all_probs = np.vstack(all_probs)
    pred_labels = np.argmax(all_probs, axis=1)
    uncertainty_probs = all_probs[:, 1]  # probability of high uncertainty

    # Metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="binary")
    report = classification_report(
        true_labels, pred_labels,
        target_names=["low_uncertainty", "high_uncertainty"],
    )
    cm = confusion_matrix(true_labels, pred_labels)

    log.info("\n=== Test Set Results ===")
    log.info("Accuracy: %.4f", accuracy)
    log.info("F1 Score: %.4f", f1)
    log.info("\nClassification Report:\n%s", report)
    log.info("\nConfusion Matrix:\n%s", cm)

    # Save metrics
    metrics_df = pd.DataFrame([{
        "accuracy": accuracy,
        "f1": f1,
        "n_test": len(test_df),
    }])
    metrics_df.to_csv(
        os.path.join(output_dir, "model_evaluation.csv"), index=False
    )

    # Show most / least uncertain sentences
    test_df = test_df.copy()
    test_df["pred_prob"] = uncertainty_probs
    test_df["pred_label"] = pred_labels

    log.info("\n=== Top 10 Most Uncertain Sentences (by model) ===")
    top_uncertain = test_df.nlargest(10, "pred_prob")
    for _, r in top_uncertain.iterrows():
        log.info("[%.3f] %s", r["pred_prob"], r["sentence"][:120])

    log.info("\n=== Top 10 Least Uncertain Sentences (by model) ===")
    least_uncertain = test_df.nsmallest(10, "pred_prob")
    for _, r in least_uncertain.iterrows():
        log.info("[%.3f] %s", r["pred_prob"], r["sentence"][:120])

    # Correlation between lexicon score and model predictions
    corr = test_df["uncertainty_ratio"].corr(test_df["pred_prob"])
    log.info("\nCorrelation (lexicon ratio vs model prob): %.4f", corr)


if __name__ == "__main__":
    main()
