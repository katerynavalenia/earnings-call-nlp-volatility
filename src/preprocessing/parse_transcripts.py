"""
Parse raw earnings call transcript .txt files into a structured parquet dataset.

Reads: scrapper/output/{year}/*.txt
Outputs: data/processed/transcripts_parsed.parquet
"""

import argparse
import os
import re
import sys
import glob
import logging
from datetime import datetime

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Patterns that indicate the Q&A section is starting.
QA_TRIGGER_PATTERNS = [
    re.compile(r"question[\s\-]*and[\s\-]*answer", re.IGNORECASE),
    re.compile(r"q\s*&\s*a\s+session", re.IGNORECASE),
    re.compile(r"open.*(?:line|floor|call).*(?:question|q&a)", re.IGNORECASE),
    re.compile(r"begin.*question", re.IGNORECASE),
    re.compile(r"take.*(?:first|your)\s+question", re.IGNORECASE),
    re.compile(r"Operator Instructions", re.IGNORECASE),
]

HEADER_SEPARATOR = "=" * 40  # actual file uses 80 '=' chars

SPEAKER_RE = re.compile(r"^(.+?)\s{2,}\[(\w[\w\s]*)\]\s*$")


def parse_metadata(lines):
    """Extract metadata key-value pairs from the header section."""
    meta = {}
    for line in lines:
        if line.startswith("="):
            break
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta


def parse_speaker_blocks(lines):
    """Split transcript body into list of (speaker, role, text) tuples."""
    blocks = []
    current_speaker = ""
    current_role = ""
    current_text_parts = []

    for line in lines:
        match = SPEAKER_RE.match(line)
        if match:
            # Save previous block
            if current_text_parts:
                text = "\n".join(current_text_parts).strip()
                if text:
                    blocks.append((current_speaker, current_role, text))
            current_speaker = match.group(1).strip()
            current_role = match.group(2).strip()
            current_text_parts = []
        else:
            current_text_parts.append(line)

    # Save last block
    if current_text_parts:
        text = "\n".join(current_text_parts).strip()
        if text:
            blocks.append((current_speaker, current_role, text))

    return blocks


def split_prepared_qa(blocks):
    """Separate prepared remarks from Q&A based on operator cues."""
    qa_start_idx = None

    for i, (speaker, role, text) in enumerate(blocks):
        if role.lower() in ("operator", "operators"):
            for pattern in QA_TRIGGER_PATTERNS:
                if pattern.search(text):
                    qa_start_idx = i
                    break
        if qa_start_idx is not None:
            break

    if qa_start_idx is not None:
        prepared = blocks[:qa_start_idx]
        qa = blocks[qa_start_idx:]
    else:
        # If no clear Q&A trigger, check if any Analyst blocks exist.
        first_analyst = None
        for i, (_, role, _) in enumerate(blocks):
            if role.lower() in ("analysts", "analyst"):
                first_analyst = i
                break
        if first_analyst is not None:
            prepared = blocks[:first_analyst]
            qa = blocks[first_analyst:]
        else:
            prepared = blocks
            qa = []

    return prepared, qa


def extract_management_text(blocks):
    """Get concatenated text from management speakers (Executives role)."""
    parts = []
    for speaker, role, text in blocks:
        if role.lower() in ("executives", "executive"):
            parts.append(text)
    return " ".join(parts)


def extract_company_name_from_filename(filename):
    """Try to pull company name from the transcript filename."""
    # Filename format: CompanyName,_Q1_2025_Earnings_Call,_Date.txt
    base = os.path.splitext(os.path.basename(filename))[0]
    # Replace underscores with spaces
    base = base.replace("_", " ")
    # Take the part before the first earnings call mention
    match = re.split(r",\s*(?:Q[1-4]|H[12]|FY|20\d{2}|Nine|2024|2025|2026)", base)
    if match:
        return match[0].strip().rstrip(",").strip()
    return base


def parse_single_file(filepath):
    """Parse one transcript file and return a dict of extracted fields."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        log.warning("Could not read %s: %s", filepath, e)
        return None

    lines = content.split("\n")

    # Find header/body boundary
    sep_idx = None
    for i, line in enumerate(lines):
        if line.startswith("=" * 40):
            sep_idx = i
            break

    if sep_idx is None:
        log.warning("No header separator found in %s", filepath)
        return None

    header_lines = lines[:sep_idx]
    body_lines = lines[sep_idx + 1:]

    meta = parse_metadata(header_lines)
    blocks = parse_speaker_blocks(body_lines)

    if not blocks:
        log.warning("No speaker blocks found in %s", filepath)
        return None

    prepared, qa = split_prepared_qa(blocks)

    full_mgmt_text = extract_management_text(blocks)
    prepared_mgmt_text = extract_management_text(prepared)
    qa_mgmt_text = extract_management_text(qa)

    # Parse event date
    event_date_raw = meta.get("Event Date", "")
    try:
        event_date = datetime.fromisoformat(
            event_date_raw.replace("Z", "+00:00")
        ).strftime("%Y-%m-%d")
    except Exception:
        event_date = ""

    company_name = extract_company_name_from_filename(filepath)

    return {
        "transcript_id": meta.get("Transcript ID", ""),
        "company_name": company_name,
        "title": meta.get("Title", ""),
        "event_date": event_date,
        "event_type": meta.get("Event Type", ""),
        "sector": meta.get("Sector", ""),
        "industry": meta.get("Industry", ""),
        "source_file": os.path.basename(filepath),
        "n_blocks": len(blocks),
        "n_prepared_blocks": len(prepared),
        "n_qa_blocks": len(qa),
        "management_text": full_mgmt_text,
        "prepared_remarks": prepared_mgmt_text,
        "qa_text": qa_mgmt_text,
    }


def find_all_transcripts(base_dir):
    """Recursively find all .txt files in output directory."""
    pattern = os.path.join(base_dir, "**", "*.txt")
    return glob.glob(pattern, recursive=True)


def main():
    parser = argparse.ArgumentParser(description="Parse earnings call transcripts.")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit the number of transcript files to parse (0 = all).",
    )
    parser.add_argument(
        "--transcript-dir", type=str, default="",
        help="Custom transcript directory (overrides default scrapper/output).",
    )
    args = parser.parse_args()

    # Resolve paths relative to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    if args.transcript_dir:
        transcript_dir = os.path.abspath(args.transcript_dir)
    else:
        transcript_dir = os.path.join(project_root, "scrapper", "output")
    output_path = os.path.join(project_root, "data", "processed", "transcripts_parsed.parquet")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    log.info("Scanning for transcripts in %s", transcript_dir)
    files = find_all_transcripts(transcript_dir)
    log.info("Found %d transcript files", len(files))

    if not files:
        log.error("No transcript files found. Check the path.")
        sys.exit(1)

    if args.limit > 0:
        files = files[: args.limit]
        log.info("Limiting to %d files", len(files))

    records = []
    errors = 0
    for fp in tqdm(files, desc="Parsing transcripts"):
        result = parse_single_file(fp)
        if result:
            records.append(result)
        else:
            errors += 1

    log.info("Parsed %d transcripts, %d errors", len(records), errors)

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("Saved to %s (%d rows)", output_path, len(df))

    # Print summary stats
    log.info("Date range: %s to %s", df["event_date"].min(), df["event_date"].max())
    log.info("Unique companies: %d", df["company_name"].nunique())
    log.info("Sectors: %s", df["sector"].value_counts().head(5).to_dict())
    log.info(
        "Avg management text length: %.0f chars",
        df["management_text"].str.len().mean(),
    )


if __name__ == "__main__":
    main()
