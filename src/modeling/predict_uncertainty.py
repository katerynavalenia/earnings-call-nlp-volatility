"""
Score all transcripts with the fine-tuned uncertainty model.

For each transcript, splits management text into sentences, runs them
through the model, and aggregates into transcript-level uncertainty scores.

Reads: data/processed/transcripts_parsed.parquet, models/uncertainty_finbert/
Outputs: data/processed/transcript_scores.parquet
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MAX_LENGTH = 128
BATCH_SIZE = 64
UNCERTAINTY_THRESHOLD = 0.5

# Import sentence splitter from labeling module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "labeling"))
from lexicon_labeler import split_into_sentences, score_sentence, UNCERTAINTY_WORDS


def predict_batch(texts, tokenizer, model, device):
    """Get uncertainty probability for a batch of texts."""
    if not texts:
        return np.array([])
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

    return probs[:, 1].cpu().numpy()  # probability of class 1 (high uncertainty)


def score_transcript_text(text, tokenizer, model, device):
    """Score a block of text and return aggregated uncertainty metrics."""
    if not text or not isinstance(text, str) or len(text.strip()) < 50:
        return {
            "uncertainty_score": np.nan,
            "uncertainty_share": np.nan,
            "n_sentences": 0,
        }

    sentences = split_into_sentences(text)
    if not sentences:
        return {
            "uncertainty_score": np.nan,
            "uncertainty_share": np.nan,
            "n_sentences": 0,
        }

    # Predict in batches
    all_probs = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i : i + BATCH_SIZE]
        probs = predict_batch(batch, tokenizer, model, device)
        all_probs.extend(probs.tolist())

    all_probs = np.array(all_probs)

    return {
        "uncertainty_score": float(np.mean(all_probs)),
        "uncertainty_share": float(np.mean(all_probs > UNCERTAINTY_THRESHOLD)),
        "n_sentences": len(sentences),
    }


def compute_lexicon_score(text):
    """Compute lexicon-based uncertainty score as a baseline."""
    if not text or not isinstance(text, str) or len(text.strip()) < 50:
        return np.nan
    sentences = split_into_sentences(text)
    if not sentences:
        return np.nan
    ratios = [score_sentence(s, UNCERTAINTY_WORDS)[0] for s in sentences]
    return float(np.mean(ratios))


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    model_dir = os.path.join(project_root, "models", "uncertainty_finbert")
    transcripts_path = os.path.join(
        project_root, "data", "processed", "transcripts_parsed.parquet"
    )
    output_path = os.path.join(
        project_root, "data", "processed", "transcript_scores.parquet"
    )

    if not os.path.exists(model_dir):
        log.error("Model not found. Run train_uncertainty_model.py first.")
        sys.exit(1)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading model (device: %s)", device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Load transcripts
    df = pd.read_parquet(transcripts_path)
    log.info("Scoring %d transcripts", len(df))

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring transcripts"):
        # Score full management text
        full_scores = score_transcript_text(
            row.get("management_text", ""), tokenizer, model, device
        )
        # Score prepared remarks
        prepared_scores = score_transcript_text(
            row.get("prepared_remarks", ""), tokenizer, model, device
        )
        # Score Q&A
        qa_scores = score_transcript_text(
            row.get("qa_text", ""), tokenizer, model, device
        )
        # Lexicon baseline
        lexicon_score = compute_lexicon_score(row.get("management_text", ""))

        results.append({
            "transcript_id": row["transcript_id"],
            "company_name": row["company_name"],
            "event_date": row["event_date"],
            "sector": row.get("sector", ""),
            "industry": row.get("industry", ""),
            "uncertainty_score": full_scores["uncertainty_score"],
            "uncertainty_share": full_scores["uncertainty_share"],
            "n_sentences": full_scores["n_sentences"],
            "prepared_uncertainty_score": prepared_scores["uncertainty_score"],
            "prepared_uncertainty_share": prepared_scores["uncertainty_share"],
            "qa_uncertainty_score": qa_scores["uncertainty_score"],
            "qa_uncertainty_share": qa_scores["uncertainty_share"],
            "lexicon_uncertainty_score": lexicon_score,
        })

    scores_df = pd.DataFrame(results)
    scores_df.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("Saved scores to %s (%d rows)", output_path, len(scores_df))

    # Summary statistics
    log.info("\n=== Uncertainty Score Summary ===")
    log.info(
        "Mean uncertainty score: %.4f",
        scores_df["uncertainty_score"].mean(),
    )
    log.info(
        "Mean uncertainty share: %.4f",
        scores_df["uncertainty_share"].mean(),
    )
    log.info(
        "Correlation (model vs lexicon): %.4f",
        scores_df["uncertainty_score"].corr(scores_df["lexicon_uncertainty_score"]),
    )


if __name__ == "__main__":
    main()
