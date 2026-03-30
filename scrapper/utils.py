"""
Helper functions for the scraper: filename cleaning, date parsing, logging.
"""

import os
import re
import logging
from datetime import datetime


def make_logger(name, log_path=None, level=logging.INFO):
    """Set up a logger that prints to console and optionally writes to a file."""
    lg = logging.getLogger(name)
    lg.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    lg.addHandler(console)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        lg.addHandler(file_handler)

    return lg


def parse_koyfin_date(raw):
    """
    Handle the various date formats Koyfin uses:
      "Feb 03 '26"  -> 2026-02-03
      "Jan 15 '09"  -> 2009-01-15
    Also tries ISO and a few other common formats as fallback.
    """
    raw = raw.strip()

    # "Mon DD 'YY" style
    m = re.match(r"([A-Za-z]{3})\s+(\d{1,2})\s+'(\d{2})", raw)
    if m:
        mon, day, yy = m.groups()
        year = int(yy)
        full_year = 2000 + year if year <= 30 else 1900 + year
        return datetime.strptime(f"{mon} {day} {full_year}", "%b %d %Y")

    # try other formats
    for pattern in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(raw, pattern)
        except ValueError:
            pass

    raise ValueError(f"Unrecognised date format: '{raw}'")


def sanitize_filename(text):
    """Strip out characters that aren't safe in filenames."""
    text = re.sub(r'[<>:"/\\|?*]', '_', text)
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_')[:200]


def make_transcript_filename(ticker, date, title):
    """
    Produce something like: AAPL_2025-01-30_Q1_2025_Earnings_Call.txt
    Keeps it under 180 chars so we don't blow up on Windows paths.
    """
    ds = date.strftime("%Y-%m-%d")
    clean_title = sanitize_filename(title)
    if clean_title.startswith(ticker):
        base = f"{ds}_{clean_title}"
    else:
        base = f"{ticker}_{ds}_{clean_title}"
    if len(base) > 180:
        base = base[:180]
    return f"{base}.txt"