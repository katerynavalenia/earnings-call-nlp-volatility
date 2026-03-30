"""
Compute post-call and pre-call stock volatility measures from daily market data.

Reads: data/market/ (per-ticker CSVs with OHLCV), data/processed/transcript_scores.parquet
Outputs: data/processed/volatility_measures.parquet
"""

import os
import sys
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Volatility windows (in trading days relative to earnings call date)
POST_CALL_WINDOW = (1, 5)    # [+1, +5] days after the call
PRE_CALL_WINDOW = (-30, -6)  # [-30, -6] days before the call

# Alternative windows for robustness
ALT_POST_WINDOWS = {
    "post_vol_1_3": (1, 3),
    "post_vol_1_10": (1, 10),
}


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


def load_market_data(market_dir, ticker):
    """Load daily market data for a single ticker."""
    fpath = find_ticker_file(market_dir, ticker)
    if fpath is None:
        return None
    try:
        if fpath.endswith((".pq", ".parquet")):
            df = pd.read_parquet(fpath)
        else:
            df = pd.read_csv(fpath)
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        # Find the date column
        date_col = None
        for candidate in ["date", "datetime", "timestamp"]:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            return None
        df["date"] = pd.to_datetime(df[date_col])
        df = df.sort_values("date").reset_index(drop=True)

        # Find close column
        close_col = None
        for candidate in ["close", "adj close", "adjclose", "adj_close"]:
            if candidate in df.columns:
                close_col = candidate
                break
        if close_col is None:
            return None

        df["log_return"] = np.log(df[close_col] / df[close_col].shift(1))
        return df[["date", close_col, "log_return"]].dropna()
    except Exception as e:
        log.debug("Error loading %s: %s", ticker, e)
        return None


def compute_realized_vol(returns):
    """Compute annualized realized volatility from a series of daily log returns."""
    if returns is None or len(returns) < 2:
        return np.nan
    return float(np.std(returns, ddof=1) * np.sqrt(252))


def compute_volatility_for_event(market_df, event_date, window_start, window_end):
    """
    Compute realized volatility for a specific window around an event.

    window_start and window_end are in trading days relative to event_date.
    Positive = after, negative = before.
    """
    if market_df is None or market_df.empty:
        return np.nan

    event_dt = pd.Timestamp(event_date)

    dates = market_df["date"].values
    event_idx = np.searchsorted(dates, np.datetime64(event_dt))

    start_idx = event_idx + window_start
    end_idx = event_idx + window_end + 1  # inclusive

    if start_idx < 0:
        start_idx = 0
    if end_idx > len(market_df):
        end_idx = len(market_df)
    if start_idx >= end_idx:
        return np.nan

    returns = market_df["log_return"].iloc[start_idx:end_idx].values
    return compute_realized_vol(returns)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    market_dir = os.path.join(project_root, "data", "market")
    market_dir = find_data_dir(market_dir)
    scores_path = os.path.join(
        project_root, "data", "processed", "transcript_scores.parquet"
    )
    ticker_map_path = os.path.join(
        project_root, "data", "processed", "ticker_map.csv"
    )
    output_path = os.path.join(
        project_root, "data", "processed", "volatility_measures.parquet"
    )

    if not os.path.exists(scores_path):
        log.error("Transcript scores not found. Run predict_uncertainty.py first.")
        sys.exit(1)
    if not os.path.exists(ticker_map_path):
        log.error("Ticker map not found. Run ticker_mapping.py first.")
        sys.exit(1)
    if not os.path.isdir(market_dir):
        log.error(
            "Market data directory not found at %s. "
            "Download and extract market.zip from the professor's drive.",
            market_dir,
        )
        sys.exit(1)

    # Load ticker map
    ticker_map = pd.read_csv(ticker_map_path)
    ticker_map = ticker_map[ticker_map["ticker"].notna() & (ticker_map["ticker"] != "")]
    company_to_ticker = dict(
        zip(ticker_map["company_name"], ticker_map["ticker"])
    )
    log.info("Ticker map: %d companies matched", len(company_to_ticker))

    # Load transcript scores
    scores_df = pd.read_parquet(scores_path)
    scores_df["ticker"] = scores_df["company_name"].map(company_to_ticker)
    matched = scores_df[scores_df["ticker"].notna()]
    log.info(
        "Transcripts with tickers: %d / %d",
        len(matched), len(scores_df),
    )

    # Cache loaded market data per ticker
    market_cache = {}

    results = []
    for _, row in tqdm(matched.iterrows(), total=len(matched), desc="Computing volatility"):
        ticker = row["ticker"]
        event_date = row["event_date"]

        if not event_date or pd.isna(event_date):
            continue

        # Load market data (cached)
        if ticker not in market_cache:
            market_cache[ticker] = load_market_data(market_dir, ticker)
        mkt = market_cache[ticker]

        # Compute primary volatility measures
        post_vol = compute_volatility_for_event(
            mkt, event_date, POST_CALL_WINDOW[0], POST_CALL_WINDOW[1]
        )
        pre_vol = compute_volatility_for_event(
            mkt, event_date, PRE_CALL_WINDOW[0], PRE_CALL_WINDOW[1]
        )

        # Abnormal volatility
        if not np.isnan(post_vol) and not np.isnan(pre_vol) and pre_vol > 0:
            abnormal_vol = post_vol - pre_vol
            vol_ratio = post_vol / pre_vol
        else:
            abnormal_vol = np.nan
            vol_ratio = np.nan

        record = {
            "transcript_id": row["transcript_id"],
            "ticker": ticker,
            "event_date": event_date,
            "post_call_vol": post_vol,
            "pre_call_vol": pre_vol,
            "abnormal_vol": abnormal_vol,
            "vol_ratio": vol_ratio,
        }

        # Alternative windows
        for name, window in ALT_POST_WINDOWS.items():
            record[name] = compute_volatility_for_event(
                mkt, event_date, window[0], window[1]
            )

        results.append(record)

    vol_df = pd.DataFrame(results)
    vol_df.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("Saved volatility measures to %s (%d rows)", output_path, len(vol_df))

    # Summary
    log.info("\n=== Volatility Summary ===")
    for col in ["post_call_vol", "pre_call_vol", "abnormal_vol"]:
        valid = vol_df[col].dropna()
        log.info(
            "%s — mean: %.4f, median: %.4f, std: %.4f, N: %d",
            col, valid.mean(), valid.median(), valid.std(), len(valid),
        )


if __name__ == "__main__":
    main()
