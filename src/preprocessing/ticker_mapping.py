"""
Map company names from transcripts to ticker symbols.

Uses the professor's market data filenames (each named by ticker) combined
with Yahoo Finance ticker-to-company-name data and fuzzy string matching.
Outputs data/processed/ticker_map.csv.
"""

import os
import re
import sys
import csv
import logging

import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Common suffixes to strip for cleaner matching
COMPANY_SUFFIXES = re.compile(
    r",?\s*\b(Inc\.?|Corp\.?|Corporation|Company|Co\.?|Ltd\.?|Limited|PLC|plc|"
    r"S\.?A\.?|S\.?A\.?B\.?\s*de\s*C\.?V\.?|AG|SE|NV|N\.V\.|"
    r"Group|Holdings?|Holding|Incorporated|Technologies|Technology)\b\.?",
    re.IGNORECASE,
)


def normalize_name(name):
    """Normalize a company name for fuzzy matching."""
    name = COMPANY_SUFFIXES.sub("", name)
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


def find_market_data_dir(base_dir):
    """Find the actual directory with per-ticker data files.

    Handles nested structure like data/market/market/ from zip extraction.
    """
    if not os.path.isdir(base_dir):
        return base_dir
    entries = os.listdir(base_dir)
    data_files = [e for e in entries if e.endswith((".csv", ".pq", ".parquet"))]
    if data_files:
        return base_dir
    subdirs = [e for e in entries if os.path.isdir(os.path.join(base_dir, e))]
    if len(subdirs) == 1:
        return find_market_data_dir(os.path.join(base_dir, subdirs[0]))
    return base_dir


def load_tickers_from_market_data(market_dir):
    """Extract ticker symbols from filenames in the market data directory."""
    tickers = []
    actual_dir = find_market_data_dir(market_dir)
    if not os.path.isdir(actual_dir):
        log.warning("Market data directory not found: %s", actual_dir)
        return tickers
    for fname in os.listdir(actual_dir):
        if fname.endswith((".csv", ".pq", ".parquet")):
            ticker = os.path.splitext(fname)[0]
            tickers.append(ticker)
    log.info("Found %d tickers from market data in %s", len(tickers), actual_dir)
    return tickers


def build_ticker_name_map(market_dir):
    """
    Build a mapping of ticker -> company name.

    Reads the first row of each market data CSV to try to extract the
    company name, or falls back to using ticker as-is.
    """
    ticker_to_name = {}
    actual_dir = find_market_data_dir(market_dir)
    tickers = load_tickers_from_market_data(market_dir)

    for ticker in tickers:
        # Try both .pq and .csv
        fpath = None
        for ext in [".pq", ".parquet", ".csv"]:
            candidate = os.path.join(actual_dir, f"{ticker}{ext}")
            if os.path.exists(candidate):
                fpath = candidate
                break
        if fpath is None:
            ticker_to_name[ticker] = ticker
            continue
        try:
            if fpath.endswith((".pq", ".parquet")):
                df = pd.read_parquet(fpath)
                df = df.head(1)
            else:
                df = pd.read_csv(fpath, nrows=1)
            # Check if there's a 'name' or 'companyName' column
            for col in ["name", "companyName", "company_name", "Name"]:
                if col in df.columns and pd.notna(df[col].iloc[0]):
                    ticker_to_name[ticker] = str(df[col].iloc[0])
                    break
            else:
                ticker_to_name[ticker] = ticker
        except Exception:
            ticker_to_name[ticker] = ticker

    return ticker_to_name


def build_known_mappings():
    """
    Hardcoded mappings for the most common US equities that are tricky
    to fuzzy-match. Extend this as needed.
    """
    return {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Alphabet": "GOOGL",
        "Amazon": "AMZN",
        "Meta Platforms": "META",
        "Tesla": "TSLA",
        "NVIDIA": "NVDA",
        "JPMorgan Chase": "JPM",
        "Johnson & Johnson": "JNJ",
        "Visa": "V",
        "Walmart": "WMT",
        "UnitedHealth Group": "UNH",
        "Procter & Gamble": "PG",
        "Mastercard": "MA",
        "Bank of America": "BAC",
        "Chevron": "CVX",
        "Home Depot": "HD",
        "Coca-Cola": "KO",
        "Merck": "MRK",
        "PepsiCo": "PEP",
        "Costco Wholesale": "COST",
        "Netflix": "NFLX",
        "Adobe": "ADBE",
        "Salesforce": "CRM",
        "Intel": "INTC",
        "Cisco Systems": "CSCO",
        "Verizon Communications": "VZ",
        "AT&T": "T",
        "Walt Disney": "DIS",
        "Goldman Sachs Group": "GS",
        "Morgan Stanley": "MS",
        "Citigroup": "C",
        "Wells Fargo": "WFC",
        "Pfizer": "PFE",
        "Berkshire Hathaway": "BRK-B",
        "Eli Lilly": "LLY",
        "AbbVie": "ABBV",
        "Broadcom": "AVGO",
        "Texas Instruments": "TXN",
        "Qualcomm": "QCOM",
        "General Motors": "GM",
        "Ford Motor": "F",
        "Boeing": "BA",
        "Caterpillar": "CAT",
        "3M": "MMM",
        "General Electric": "GE",
        "Honeywell International": "HON",
        "Lockheed Martin": "LMT",
        "United Parcel Service": "UPS",
        "Starbucks": "SBUX",
        "McDonald's": "MCD",
        "Nike": "NKE",
        "Target": "TGT",
        "Lowe's": "LOW",
    }


def map_companies_to_tickers(
    company_names, market_dir, score_threshold=75
):
    """
    Map a list of company names to ticker symbols using:
    1. Known hardcoded mappings
    2. Exact match against ticker-to-name map
    3. Fuzzy matching
    """
    known = build_known_mappings()
    known_norm = {normalize_name(k): v for k, v in known.items()}

    ticker_to_name = build_ticker_name_map(market_dir)
    # Build reverse map: normalized name -> ticker
    name_to_ticker = {}
    for ticker, name in ticker_to_name.items():
        name_to_ticker[normalize_name(name)] = ticker
        # Also add the raw ticker itself as a possible match
        name_to_ticker[ticker.lower()] = ticker

    # All candidate names for fuzzy matching
    all_candidates = list(name_to_ticker.keys()) + list(known_norm.keys())

    results = []
    unmatched = []

    for company in tqdm(company_names, desc="Mapping tickers"):
        norm = normalize_name(company)

        # 1. Check known mappings
        if norm in known_norm:
            results.append({
                "company_name": company,
                "ticker": known_norm[norm],
                "match_method": "known",
                "match_score": 100,
            })
            continue

        # 2. Check exact match in name_to_ticker
        if norm in name_to_ticker:
            results.append({
                "company_name": company,
                "ticker": name_to_ticker[norm],
                "match_method": "exact",
                "match_score": 100,
            })
            continue

        # 3. Fuzzy match
        match = process.extractOne(
            norm, all_candidates, scorer=fuzz.token_sort_ratio
        )
        if match and match[1] >= score_threshold:
            matched_name = match[0]
            if matched_name in name_to_ticker:
                ticker = name_to_ticker[matched_name]
            elif matched_name in known_norm:
                ticker = known_norm[matched_name]
            else:
                ticker = None

            if ticker:
                results.append({
                    "company_name": company,
                    "ticker": ticker,
                    "match_method": "fuzzy",
                    "match_score": match[1],
                })
                continue

        # No match found
        unmatched.append(company)
        results.append({
            "company_name": company,
            "ticker": "",
            "match_method": "unmatched",
            "match_score": 0,
        })

    return results, unmatched


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    transcripts_path = os.path.join(
        project_root, "data", "processed", "transcripts_parsed.parquet"
    )
    market_dir = os.path.join(project_root, "data", "market")
    output_path = os.path.join(project_root, "data", "processed", "ticker_map.csv")

    if not os.path.exists(transcripts_path):
        log.error(
            "Parsed transcripts not found at %s. Run parse_transcripts.py first.",
            transcripts_path,
        )
        sys.exit(1)

    df = pd.read_parquet(transcripts_path)
    company_names = df["company_name"].dropna().unique().tolist()
    log.info("Unique company names to map: %d", len(company_names))

    results, unmatched = map_companies_to_tickers(company_names, market_dir)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    matched = results_df[results_df["ticker"] != ""]
    log.info(
        "Matched: %d / %d (%.1f%%)",
        len(matched),
        len(results_df),
        100 * len(matched) / max(len(results_df), 1),
    )
    log.info("Match methods: %s", matched["match_method"].value_counts().to_dict())

    if unmatched:
        unmatched_path = os.path.join(
            project_root, "data", "processed", "unmatched_companies.txt"
        )
        with open(unmatched_path, "w", encoding="utf-8") as f:
            for name in sorted(unmatched):
                f.write(name + "\n")
        log.info("Unmatched companies saved to %s", unmatched_path)

    log.info("Ticker map saved to %s", output_path)


if __name__ == "__main__":
    main()
