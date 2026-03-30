"""
Runner for the Koyfin transcript scraper.

Can run as a single process or split the work across multiple browsers
processing different months in parallel.

"""

import os
import sys
import argparse
import time
import logging
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---- config / env ----

def _read_dotenv():
    """Read .env file next to this script if it exists."""
    dotenv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(dotenv):
        return
    with open(dotenv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

_read_dotenv()

EMAIL = os.environ.get("KOYFIN_EMAIL", "")
PASSWORD = os.environ.get("KOYFIN_PASSWORD", "")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
PROGRESS_DIR = os.path.join(OUT_DIR, "progress")
LOG_DIR = os.path.join(OUT_DIR, "errors")


def _make_logger(name, log_path=None):
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    lg.addHandler(sh)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    return lg


# ---- split date range into months ----

def _monthly_chunks(start, end):
    """Break a date range into per-month (start, end) pairs as YYYY-MM-DD strings."""
    result = []
    cur = start.replace(day=1)
    while cur <= end:
        chunk_start = max(cur, start)
        if cur.month == 12:
            nxt = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            nxt = cur.replace(month=cur.month + 1, day=1)
        chunk_end = min(nxt - timedelta(days=1), end)
        if chunk_start <= chunk_end:
            result.append((chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cur = nxt
    return result


# ---- worker (runs in its own process) ----

def _process_chunk(start_date, end_date, batch_size, retries=2):
    """Scrape a single date chunk. Meant to run in a subprocess."""
    _read_dotenv()
    email = os.environ.get("KOYFIN_EMAIL", "")
    pwd = os.environ.get("KOYFIN_PASSWORD", "")

    tag = f"{start_date}_{end_date}"
    log_path = os.path.join(LOG_DIR, f"chunk_{tag}.log")
    checkpoint = os.path.join(PROGRESS_DIR, f"chunk_{tag}.json")
    logger = _make_logger(f"chunk_{tag}", log_path)

    t0 = time.time()
    n_saved = 0

    for attempt in range(retries + 1):
        scraper = None
        try:
            from koyfin_scraper import TranscriptScraper

            scraper = TranscriptScraper(
                email=email,
                password=pwd,
                headless=True,
                logger=logger,
            )
            scraper.start()
            n_saved = scraper.scrape_range(
                start_date=start_date,
                end_date=end_date,
                dest=OUT_DIR,
                resume_file=checkpoint,
                batch_size=batch_size,
            )
            scraper.stop()
            break

        except Exception as e:
            logger.error("Attempt %d failed: %s", attempt + 1, str(e)[:200])
            if scraper:
                try:
                    scraper.stop()
                except Exception:
                    pass
            if attempt < retries:
                time.sleep(min(30 * (attempt + 1), 120))
            else:
                logger.error("Gave up on %s after %d attempts", tag, retries + 1)

    mins = round((time.time() - t0) / 60, 1)
    return {"tag": tag, "saved": n_saved, "minutes": mins}


# ---- dry run ----

def _preview(start_date, end_date):
    """Just show how many transcripts exist per month, don't download anything."""
    from koyfin_scraper import TranscriptScraper

    scraper = TranscriptScraper(email=EMAIL, password=PASSWORD, headless=True)
    scraper.start()

    months = _monthly_chunks(
        datetime.strptime(start_date, "%Y-%m-%d"),
        datetime.strptime(end_date, "%Y-%m-%d"),
    )

    grand_total = 0
    print(f"\n{'Period':<25} {'Count':>15}")
    print("-" * 42)

    for s, e in months:
        iso_s = f"{s}T00:00:00.000Z"
        iso_e = f"{e}T23:59:59.000Z"
        page = scraper._run_search(iso_s, iso_e, 0)
        n = page.get("totalHits", 0)
        grand_total += n
        print(f"  {s} — {e}    {n:>8}")

    print("-" * 42)
    print(f"  {'TOTAL':<22} {grand_total:>8}")
    print(f"\n  Rough ETA with 5 workers: ~{grand_total * 0.3 / 60 / 5:.0f} min")
    scraper.stop()


# ---- entry point ----

def main():
    ap = argparse.ArgumentParser(description="Download Koyfin earnings call transcripts")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel browsers (default: 1)")
    ap.add_argument("--batch-size", type=int, default=5, help="Transcripts per JS call (default: 5)")
    ap.add_argument("--dry-run", action="store_true", help="Only show counts, don't download")
    args = ap.parse_args()

    if not EMAIL or not PASSWORD:
        print("ERROR: KOYFIN_EMAIL and KOYFIN_PASSWORD must be set in .env")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.dry_run:
        _preview(args.start, args.end)
        return

    dt_start = datetime.strptime(args.start, "%Y-%m-%d")
    dt_end = datetime.strptime(args.end, "%Y-%m-%d")

    # single worker — just run directly, no subprocess overhead
    if args.workers <= 1:
        res = _process_chunk(args.start, args.end, args.batch_size)
        print(f"\nDone: {res['saved']} saved in {res['minutes']} min")
        return

    # parallel mode
    months = _monthly_chunks(dt_start, dt_end)

    print(f"\n{'=' * 55}")
    print(f"Koyfin Transcript Scraper")
    print(f"Range:     {args.start} to {args.end}")
    print(f"Chunks:    {len(months)} months")
    print(f"Workers:   {args.workers}")
    print(f"Batch:     {args.batch_size} per call")
    print(f"{'=' * 55}")

    total_saved = 0
    total_failed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(_process_chunk, s, e, args.batch_size): (s, e)
            for s, e in months
        }

        for future in as_completed(jobs):
            s, e = jobs[future]
            try:
                res = future.result()
                total_saved += res["saved"]
                print(f"  [OK]   {res['tag']}  saved={res['saved']}  ({res['minutes']} min)")
            except Exception as exc:
                total_failed += 1
                print(f"  [FAIL] {s}_{e}: {exc}")

    total_mins = round((time.time() - t_start) / 60, 1)
    print(f"\n{'=' * 55}")
    print(f"DONE — {total_saved} transcripts, {total_failed} failed chunks, {total_mins} min")
    print(f"Output: {os.path.abspath(OUT_DIR)}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()