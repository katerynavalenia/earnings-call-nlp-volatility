"""
Build a balanced training set from lexicon-scored sentences for FinBERT
fine-tuning.

Reads: data/processed/sentences_scored.parquet
Outputs: data/processed/train.parquet, val.parquet, test.parquet
"""

import os
import sys
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Target counts per class
TARGET_PER_CLASS = 15000
RANDOM_SEED = 42


def main():
    import pyarrow.parquet as pq

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    scored_path = os.path.join(
        project_root, "data", "processed", "sentences_scored.parquet"
    )
    output_dir = os.path.join(project_root, "data", "processed")

    if not os.path.exists(scored_path):
        log.error("Scored sentences not found. Run lexicon_labeler.py first.")
        sys.exit(1)

    needed_cols = ["sentence", "label", "n_tokens", "transcript_id", "sector",
                   "event_date", "uncertainty_ratio"]
    pf = pq.ParquetFile(scored_path)
    log.info("Reading scored sentences in chunks (total rows: %d)...", pf.metadata.num_rows)

    high_chunks = []
    low_chunks = []
    BATCH = 500_000

    for batch in pf.iter_batches(batch_size=BATCH, columns=needed_cols):
        chunk = batch.to_pandas()
        h = chunk[(chunk["label"] == "high_uncertainty") & chunk["n_tokens"].between(5, 200)]
        l = chunk[(chunk["label"] == "low_uncertainty") & chunk["n_tokens"].between(5, 200)]
        if len(h) > 0:
            high_chunks.append(h)
        if len(l) > 0:
            low_chunks.append(l)
        del chunk
        log.info("Read batch — high: %d, low: %d so far",
                 sum(len(c) for c in high_chunks), sum(len(c) for c in low_chunks))

    high = pd.concat(high_chunks, ignore_index=True)
    low = pd.concat(low_chunks, ignore_index=True)
    del high_chunks, low_chunks
    log.info("High uncertainty sentences: %d", len(high))
    log.info("Low uncertainty sentences: %d", len(low))

    # Sample to balance classes
    n_samples = min(TARGET_PER_CLASS, len(high), len(low))
    log.info("Sampling %d per class (total: %d)", n_samples, n_samples * 2)

    high_sample = high.sample(n=n_samples, random_state=RANDOM_SEED)
    low_sample = low.sample(n=n_samples, random_state=RANDOM_SEED)

    balanced = pd.concat([high_sample, low_sample], ignore_index=True)

    # Binary label: 1 = high uncertainty, 0 = low uncertainty
    balanced["binary_label"] = (balanced["label"] == "high_uncertainty").astype(int)

    # Keep only the columns needed for training
    balanced = balanced[
        ["sentence", "binary_label", "transcript_id", "sector", "event_date",
         "uncertainty_ratio"]
    ].copy()

    # Stratified split: 80% train, 10% val, 10% test
    train_df, temp_df = train_test_split(
        balanced, test_size=0.2, random_state=RANDOM_SEED,
        stratify=balanced["binary_label"],
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED,
        stratify=temp_df["binary_label"],
    )

    log.info("Train: %d, Val: %d, Test: %d", len(train_df), len(val_df), len(test_df))
    log.info(
        "Train label distribution:\n%s",
        train_df["binary_label"].value_counts().to_string(),
    )

    # Save
    train_df.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(output_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)

    log.info("Saved train/val/test splits to %s", output_dir)


if __name__ == "__main__":
    main()
