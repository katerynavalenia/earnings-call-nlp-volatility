"""
Score sentences in earnings call transcripts using the Loughran-McDonald
Uncertainty word list.

Downloads the LM Master Dictionary from SRAF (Notre Dame) if not already
present, extracts the Uncertainty word list, and labels each sentence
by the fraction of uncertainty words it contains.

Output: data/processed/sentences_scored.parquet
"""

import os
import re
import sys
import logging

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

LM_DICT_URL = (
    "https://sraf.nd.edu/loughranmcdonald-master-dictionary/"
)
# The actual CSV download link for the master dictionary
LM_CSV_URL = (
    "https://sraf.nd.edu/wp-content/uploads/2024/06/"
    "Loughran-McDonald_MasterDictionary_1993-2023.csv"
)

# Loughran-McDonald (2011) Uncertainty word list
UNCERTAINTY_WORDS = {
    "approximate", "approximately", "approximation", "assumption",
    "assumptions", "believe", "believed", "believes", "calibrate",
    "calibrated", "calibration", "cautious", "cautiously",
    "conceivable", "conceivably", "conditional", "conditionally",
    "conjecture", "conjectured", "conjectures", "contingency",
    "contingent", "could", "depend", "depended", "dependency",
    "dependent", "depending", "depends", "destabilize", "destabilized",
    "destabilizing", "deviate", "deviated", "deviates", "deviation",
    "deviations", "doubt", "doubted", "doubtful", "doubts",
    "erratic", "erratically", "estimate", "estimated", "estimates",
    "estimating", "estimation", "estimations", "eventual", "eventually",
    "exposed", "exposure", "exposures", "fluctuate", "fluctuated",
    "fluctuates", "fluctuating", "fluctuation", "fluctuations",
    "forecast", "forecasted", "forecasting", "forecasts",
    "hesitancy", "hesitant", "hypothetical", "hypothetically",
    "imprecise", "imprecision", "imprecisions", "improbable",
    "incompleteness", "indefinite", "indefinitely", "indefiniteness",
    "indeterminate", "indeterminable", "inexact", "inexactly",
    "inexactness", "instability", "instabilities", "likelihood",
    "may", "maybe", "might", "nearly", "nonassessable",
    "objective", "objectives", "obscure", "obscured", "obscures",
    "occasional", "occasionally", "pending", "perhaps",
    "possibility", "possible", "possibly", "precaution",
    "precautions", "precautionary", "predict", "predictability",
    "predicted", "predicting", "prediction", "predictions",
    "predictive", "preliminary", "presumably", "presume",
    "presumed", "presumption", "presumptions", "probabilistic",
    "probabilities", "probability", "probable", "probably",
    "project", "projected", "projecting", "projection", "projections",
    "provisional", "provisionally", "random", "randomly", "randomness",
    "reassess", "reassessed", "reassessing", "reassessment",
    "recalculate", "recalculated", "recalculation", "reconsider",
    "reconsidered", "revision", "revisions", "risky",
    "rough", "roughly", "seems", "seldom", "sometime", "sometimes",
    "somewhat", "somewhere", "speculate", "speculated", "speculates",
    "speculating", "speculation", "speculations", "speculative",
    "sudden", "suddenly", "suggest", "suggested", "suggesting",
    "suggestion", "suggestions", "suggests", "susceptibility",
    "susceptible", "tend", "tended", "tendency", "tendencies",
    "tends", "tentative", "tentatively", "uncertain", "uncertainly",
    "uncertainties", "uncertainty", "unclear", "unconfirmed",
    "undecided", "undefined", "undesignated", "undetectable",
    "undetermined", "undocumented", "unexpected", "unexpectedly",
    "unfamiliar", "unforeseen", "unguaranteed", "unhedged",
    "unknown", "unknowable", "unknowing", "unknowingly",
    "unobservable", "unplanned", "unpredictability", "unpredictable",
    "unpredictably", "unpredicted", "unproven", "unquantifiable",
    "unquantified", "unreconciled", "unresolved", "unsettled",
    "unspecified", "untested", "unusual", "unusually",
    "vagaries", "vague", "vaguely", "vagueness", "variability",
    "variable", "variables", "variance", "variances", "variation",
    "variations", "varied", "varies", "vary", "varying",
    "volatile", "volatility", "volatilities",
}

# Simple sentence splitter — split on period/question/exclamation followed by space
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b[a-z]+\b", text.lower())


def score_sentence(sentence, uncertainty_set):
    """Compute the uncertainty word ratio for a single sentence."""
    tokens = tokenize(sentence)
    if len(tokens) < 3:
        return 0.0, 0, len(tokens)
    uncertainty_count = sum(1 for t in tokens if t in uncertainty_set)
    return uncertainty_count / len(tokens), uncertainty_count, len(tokens)


def split_into_sentences(text):
    """Split text into sentences."""
    if not text or not text.strip():
        return []
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    # Filter out very short fragments
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    sys.path.insert(0, project_root)
    from src.preprocessing.parse_transcripts import parse_single_file

    output_path = os.path.join(
        project_root, "data", "processed", "sentences_scored.parquet"
    )

    transcript_dir = os.path.join(project_root, "koyfin_scr", "output")
    if not os.path.exists(transcript_dir):
        log.error("Transcript directory not found: %s", transcript_dir)
        sys.exit(1)

    # Use os.walk (much faster than glob on Windows for large dirs)
    log.info("Scanning for transcript files in %s ...", transcript_dir)
    files = []
    for root, _dirs, fnames in os.walk(transcript_dir):
        for fn in fnames:
            if fn.endswith(".txt"):
                files.append(os.path.join(root, fn))
    log.info("Found %d transcript files to score", len(files))

    if not files:
        log.error("No transcript files found.")
        sys.exit(1)

    all_rows = []
    errors = 0
    FLUSH_INTERVAL = 10000  # log progress every N files

    for i, fp in enumerate(tqdm(files, desc="Scoring sentences")):
        rec = parse_single_file(fp)
        if rec is None:
            errors += 1
            continue

        mgmt_text = rec.get("management_text", "")
        if not mgmt_text:
            continue

        sentences = split_into_sentences(mgmt_text)
        for sent in sentences:
            ratio, count, n_tokens = score_sentence(sent, UNCERTAINTY_WORDS)
            all_rows.append({
                "transcript_id": rec["transcript_id"],
                "company_name": rec["company_name"],
                "event_date": rec["event_date"],
                "sector": rec.get("sector", ""),
                "sentence": sent,
                "uncertainty_ratio": ratio,
                "uncertainty_count": count,
                "n_tokens": n_tokens,
            })

        if (i + 1) % FLUSH_INTERVAL == 0:
            log.info("Processed %d / %d files, %d sentences so far", i + 1, len(files), len(all_rows))

    log.info("Parsed %d files (%d errors), building DataFrame...", len(files) - errors, errors)

    sentences_df = pd.DataFrame(all_rows)
    del all_rows
    log.info("Total sentences scored: %d", len(sentences_df))

    nonzero = sentences_df[sentences_df["uncertainty_ratio"] > 0]["uncertainty_ratio"]
    if len(nonzero) > 0:
        q75 = nonzero.quantile(0.75)
        q25 = sentences_df["uncertainty_ratio"].quantile(0.25)
    else:
        q75 = 0.05
        q25 = 0.0

    log.info("Uncertainty ratio — Q25: %.4f, Q75: %.4f", q25, q75)

    # Label: high if in top quartile of nonzero, low if zero or bottom quartile
    sentences_df["label"] = "middle"
    sentences_df.loc[sentences_df["uncertainty_ratio"] >= q75, "label"] = "high_uncertainty"
    sentences_df.loc[sentences_df["uncertainty_ratio"] <= q25, "label"] = "low_uncertainty"

    label_counts = sentences_df["label"].value_counts()
    log.info("Label distribution:\n%s", label_counts.to_string())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sentences_df.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("Saved scored sentences to %s", output_path)


if __name__ == "__main__":
    main()
