"""
Build the final panel dataset by merging transcript uncertainty scores,
volatility measures, and control variables.

Reads:
  - data/processed/transcript_scores.parquet
  - data/processed/volatility_measures.parquet
  - data/processed/ticker_map.csv
  - data/marketCap/ (daily market cap CSVs)
  - data/earning/ (quarterly earnings data CSVs)

Outputs: data/processed/panel_dataset.parquet
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def find_data_dir(base_dir):
    """Find actual directory with per-ticker data files (handles nested zip extraction)."""
    if not os.path.isdir(base_dir):
        return base_dir
    entries = os.listdir(base_dir)
    data_files = [e for e in entries if e.endswith((".csv", ".pq", ".parquet"))]
    if data_files:
        return base_dir
    subdirs = [e for e in entries if os.path.isdir(os.path.join(base_dir, e))]
    if len(subdirs) == 1:
        return find_data_dir(os.path.join(base_dir, subdirs[0]))
    return base_dir


def find_ticker_file(data_dir, ticker):
    """Find the data file for a ticker, trying .pq, .parquet, .csv extensions."""
    for ext in [".pq", ".parquet", ".csv"]:
        fpath = os.path.join(data_dir, f"{ticker}{ext}")
        if os.path.exists(fpath):
            return fpath
    return None


def load_market_cap(marketcap_dir, ticker, event_date):
    """Load market cap for a ticker on or just before the event date."""
    fpath = find_ticker_file(marketcap_dir, ticker)
    if fpath is None:
        return np.nan
    try:
        if fpath.endswith((".pq", ".parquet")):
            df = pd.read_parquet(fpath)
        else:
            df = pd.read_csv(fpath)
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next((c for c in ["date", "datetime"] if c in df.columns), None)
        if date_col is None:
            return np.nan
        df["date"] = pd.to_datetime(df[date_col])

        mcap_col = None
        for candidate in ["marketcap", "market_cap", "marketcapitalization", "value"]:
            if candidate in df.columns:
                mcap_col = candidate
                break
        if mcap_col is None:
            # Use the first numeric column that's not date-related
            for c in df.columns:
                if c not in ["date", date_col] and pd.api.types.is_numeric_dtype(df[c]):
                    mcap_col = c
                    break
        if mcap_col is None:
            return np.nan

        event_dt = pd.Timestamp(event_date)
        df = df[df["date"] <= event_dt].sort_values("date")
        if df.empty:
            return np.nan
        return float(df[mcap_col].iloc[-1])
    except Exception:
        return np.nan


def load_earnings_surprise(earning_dir, ticker, event_date):
    """Load earnings surprise for a ticker around the event date."""
    fpath = find_ticker_file(earning_dir, ticker)
    if fpath is None:
        return np.nan
    try:
        if fpath.endswith((".pq", ".parquet")):
            df = pd.read_parquet(fpath)
        else:
            df = pd.read_csv(fpath)
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next((c for c in ["date", "datetime", "reportdate"] if c in df.columns), None)
        if date_col is None:
            return np.nan
        df["date"] = pd.to_datetime(df[date_col])

        event_dt = pd.Timestamp(event_date)
        # Find the closest earnings report on or before the event
        df = df[df["date"] <= event_dt + pd.Timedelta(days=3)]
        df = df[df["date"] >= event_dt - pd.Timedelta(days=30)]
        if df.empty:
            return np.nan

        df = df.sort_values("date").iloc[-1]

        # Look for actual and estimate columns
        actual_col = None
        estimate_col = None
        for c in ["actualeps", "actual_eps", "actual", "epsactual"]:
            if c in df.index:
                actual_col = c
                break
        for c in ["estimatedeps", "estimated_eps", "estimate", "epsestimate",
                   "epsestimated", "forecasteps", "forecast_eps", "consensus"]:
            if c in df.index:
                estimate_col = c
                break

        if actual_col is None or estimate_col is None:
            return np.nan

        actual = float(df[actual_col])
        estimate = float(df[estimate_col])
        if abs(estimate) < 0.001:
            return np.nan
        return (actual - estimate) / abs(estimate)
    except Exception:
        return np.nan


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    scores_path = os.path.join(project_root, "data", "processed", "transcript_scores.parquet")
    vol_path = os.path.join(project_root, "data", "processed", "volatility_measures.parquet")
    ticker_map_path = os.path.join(project_root, "data", "processed", "ticker_map.csv")
    marketcap_dir = os.path.join(project_root, "data", "marketCap")
    marketcap_dir = find_data_dir(marketcap_dir)
    earning_dir = os.path.join(project_root, "data", "earning")
    earning_dir = find_data_dir(earning_dir)
    output_path = os.path.join(project_root, "data", "processed", "panel_dataset.parquet")

    # Load core data
    scores_df = pd.read_parquet(scores_path)
    vol_df = pd.read_parquet(vol_path)
    ticker_map = pd.read_csv(ticker_map_path)
    ticker_map = ticker_map[ticker_map["ticker"].notna() & (ticker_map["ticker"] != "")]
    company_to_ticker = dict(zip(ticker_map["company_name"], ticker_map["ticker"]))

    log.info("Transcript scores: %d", len(scores_df))
    log.info("Volatility measures: %d", len(vol_df))

    # Add ticker to scores
    scores_df["ticker"] = scores_df["company_name"].map(company_to_ticker)

    # Merge scores with volatility on transcript_id
    panel = scores_df.merge(
        vol_df[["transcript_id", "post_call_vol", "pre_call_vol",
                "abnormal_vol", "vol_ratio", "post_vol_1_3", "post_vol_1_10"]],
        on="transcript_id",
        how="inner",
    )
    log.info("After merging scores + volatility: %d rows", len(panel))

    # Add control variables
    has_marketcap = os.path.isdir(marketcap_dir)
    has_earnings = os.path.isdir(earning_dir)

    if has_marketcap:
        log.info("Loading market cap data...")
        mcap_cache = {}
        market_caps = []
        for _, row in tqdm(panel.iterrows(), total=len(panel), desc="Market cap"):
            ticker = row["ticker"]
            event_date = row["event_date"]
            cache_key = (ticker, event_date)
            if cache_key not in mcap_cache:
                mcap_cache[cache_key] = load_market_cap(
                    marketcap_dir, ticker, event_date
                )
            market_caps.append(mcap_cache[cache_key])
        panel["market_cap"] = market_caps
        panel["log_market_cap"] = np.log(panel["market_cap"].replace(0, np.nan))
    else:
        log.warning("Market cap directory not found. Skipping market cap controls.")
        panel["market_cap"] = np.nan
        panel["log_market_cap"] = np.nan

    if has_earnings:
        log.info("Loading earnings surprise data...")
        earn_cache = {}
        surprises = []
        for _, row in tqdm(panel.iterrows(), total=len(panel), desc="Earnings"):
            ticker = row["ticker"]
            event_date = row["event_date"]
            cache_key = (ticker, event_date)
            if cache_key not in earn_cache:
                earn_cache[cache_key] = load_earnings_surprise(
                    earning_dir, ticker, event_date
                )
            surprises.append(earn_cache[cache_key])
        panel["earnings_surprise"] = surprises
    else:
        log.warning("Earnings directory not found. Skipping earnings surprise.")
        panel["earnings_surprise"] = np.nan

    # Add time fixed effects
    panel["event_date_dt"] = pd.to_datetime(panel["event_date"])
    panel["year"] = panel["event_date_dt"].dt.year
    panel["quarter"] = panel["event_date_dt"].dt.quarter
    panel["year_quarter"] = (
        panel["year"].astype(str) + "Q" + panel["quarter"].astype(str)
    )

    # Drop rows with missing key variables
    key_cols = ["uncertainty_score", "post_call_vol"]
    before = len(panel)
    panel = panel.dropna(subset=key_cols)
    log.info("Dropped %d rows with missing key variables", before - len(panel))

    # Save
    panel.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("Panel dataset saved to %s (%d rows)", output_path, len(panel))

    # Summary
    log.info("\n=== Panel Summary ===")
    log.info("Unique tickers: %d", panel["ticker"].nunique())
    log.info("Date range: %s to %s", panel["event_date"].min(), panel["event_date"].max())
    log.info("Sectors: %d", panel["sector"].nunique())
    for col in ["uncertainty_score", "uncertainty_share", "post_call_vol",
                "pre_call_vol", "abnormal_vol"]:
        if col in panel.columns:
            valid = panel[col].dropna()
            log.info(
                "%s — mean: %.4f, std: %.4f, N: %d",
                col, valid.mean(), valid.std(), len(valid),
            )


if __name__ == "__main__":
    main()
