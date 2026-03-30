"""
Koyfin transcript downloader.

Grabs earnings call transcripts through Koyfin's internal API endpoints
instead of trying to interact with the DOM (way faster).

Selenium is only used for the initial login to grab the JWT,
then everything runs as JS fetch() calls inside that browser session.
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException

from utils import sanitize_filename

# can be set from outside to gracefully stop the scraper
halt_flag = threading.Event()


# --------------- JS snippets executed in the browser ---------------

_JS_EXTRACT_TOKEN = """
var cookies = document.cookie.split(';');
for (var i = 0; i < cookies.length; i++) {
    var c = cookies[i].trim();
    if (c.indexOf('auth_token=') === 0) {
        return c.substring('auth_token='.length);
    }
}
return null;
"""

_JS_RUN_SEARCH = """
var tok = arguments[0];
var body = arguments[1];

var r = await fetch('https://app.koyfin.com/api/v1/pubhub/transcript/search', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + tok
    },
    body: JSON.stringify(body),
    credentials: 'include'
});

if (r.status === 401) return JSON.stringify({error: 'auth_expired', status: 401});
if (r.status === 429) return JSON.stringify({error: 'rate_limited', status: 429});
if (!r.ok) return JSON.stringify({error: 'http_' + r.status, status: r.status});

var j = await r.json();
return JSON.stringify({
    totalHits: j.totalHits,
    hits: (j.hits || []).map(function(item) {
        return {
            transcriptId: item.transcriptId,
            KID: item.KID,
            sector: item.sector,
            industry: item.industry,
            eventType: item.eventType,
            transcriptTitle: item.transcriptTitle,
            eventDateTime: item.eventDateTime,
            createdAt: item.createdAt
        };
    })
});
"""

_JS_FETCH_SINGLE = """
var tok = arguments[0];
var tid = arguments[1];

var r = await fetch('https://app.koyfin.com/api/v1/pubhub/transcript/' + tid, {
    headers: {'Authorization': 'Bearer ' + tok},
    credentials: 'include'
});

if (r.status === 401) return JSON.stringify({error: 'auth_expired'});
if (r.status === 429) return JSON.stringify({error: 'rate_limited'});
if (!r.ok) return JSON.stringify({error: 'http_' + r.status});

var j = await r.json();
var hdr = j.header || {};
var parts = j.components || [];

return JSON.stringify({
    header: {
        transcriptTitle: hdr.transcriptTitle || '',
        eventType: hdr.eventType || '',
        eventDateTime: hdr.eventDateTime || '',
        announcedDate: hdr.announcedDate || '',
        duration: hdr.duration || 0,
        formattedTitle: hdr.formattedTitle || '',
        title: hdr.title || ''
    },
    components: parts.map(function(p) {
        return {
            order: p.componentOrder,
            speaker: p.speakerName || '',
            role: p.speakerType || '',
            text: p.text || ''
        };
    })
});
"""

# batch version — pulls multiple transcripts in one JS execution
_JS_FETCH_MANY = """
var tok = arguments[0];
var idList = arguments[1];
var out = [];

for (var i = 0; i < idList.length; i++) {
    try {
        var r = await fetch('https://app.koyfin.com/api/v1/pubhub/transcript/' + idList[i], {
            headers: {'Authorization': 'Bearer ' + tok},
            credentials: 'include'
        });

        if (r.status === 401) { out.push({id: idList[i], error: 'auth_expired'}); continue; }
        if (r.status === 429) { out.push({id: idList[i], error: 'rate_limited'}); break; }
        if (!r.ok) { out.push({id: idList[i], error: 'http_' + r.status}); continue; }

        var j = await r.json();
        var hdr = j.header || {};
        var parts = j.components || [];

        out.push({
            id: idList[i],
            header: {
                transcriptTitle: hdr.transcriptTitle || '',
                eventType: hdr.eventType || '',
                eventDateTime: hdr.eventDateTime || '',
                announcedDate: hdr.announcedDate || '',
                duration: hdr.duration || 0,
                formattedTitle: hdr.formattedTitle || '',
                title: hdr.title || ''
            },
            components: parts.map(function(p) {
                return {
                    order: p.componentOrder,
                    speaker: p.speakerName || '',
                    role: p.speakerType || '',
                    text: p.text || ''
                };
            })
        });
    } catch(err) {
        out.push({id: idList[i], error: err.message});
    }
}

return JSON.stringify(out);
"""


class TranscriptScraper:
    """
    Downloads earnings call transcripts from Koyfin's internal API.

    Example:
        s = TranscriptScraper(email, pwd)
        s.start()
        n = s.scrape_range('2025-01-01', '2025-12-31', dest='output')
        s.stop()
    """

    _LOGIN_PAGE = "https://app.koyfin.com/login"
    _PER_PAGE = 100
    _HARD_CAP = 1000   # koyfin only gives you pages 0 through 9

    def __init__(self, email, password, headless=True, logger=None):
        self.email = email
        self.password = password
        self.headless = headless
        self.log = logger or logging.getLogger("transcript_scraper")
        self.driver: Optional[webdriver.Chrome] = None
        self._jwt: Optional[str] = None

    # ---- browser setup / teardown ----

    def start(self):
        """Spin up Chrome and log in to Koyfin."""
        opts = Options()
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--window-size=1280,720")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        # skip loading images, we don't need them
        opts.add_experimental_option("prefs", {
            "profile.managed_default_content_settings.images": 2,
        })
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])

        try:
            self.driver = webdriver.Chrome(options=opts)
        except WebDriverException:
            from webdriver_manager.chrome import ChromeDriverManager
            svc = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=svc, options=opts)

        self.driver.implicitly_wait(2)
        self._do_login()
        self.log.info("Browser ready, logged in")

    def stop(self):
        """Shut down the browser."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
            self._jwt = None

    def _do_login(self):
        """Fill in the login form and grab the JWT from cookies."""
        self.driver.get(self._LOGIN_PAGE)
        time.sleep(2)

        wait = WebDriverWait(self.driver, 15)

        # email field
        email_field = wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "input[type='email'], input[name='email']")
            )
        )
        email_field.clear()
        email_field.send_keys(self.email)
        time.sleep(0.3)

        # password field
        pwd_field = self.driver.find_element(By.CSS_SELECTOR, "input[type='password']")
        pwd_field.clear()
        pwd_field.send_keys(self.password)
        time.sleep(0.3)

        # click submit or press enter
        try:
            submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            submit_btn.click()
        except Exception:
            pwd_field.send_keys(Keys.RETURN)

        # wait for page to settle
        time.sleep(8)

        # dismiss cookie banner if it shows up
        try:
            self.driver.find_element(
                By.CSS_SELECTOR, 'button[data-cky-tag="accept-button"]'
            ).click()
        except Exception:
            pass

        # read token from cookies
        self._jwt = self.driver.execute_script(_JS_EXTRACT_TOKEN)
        if not self._jwt:
            raise RuntimeError("Login failed — couldn't find auth_token cookie")
        self.log.info("Login successful, got token")

    def _relogin(self):
        """Re-authenticate when the token expires."""
        self.log.info("Token expired, logging in again...")
        self.driver.get(self._LOGIN_PAGE)
        time.sleep(2)
        self._do_login()

    # ---- API calls (run as JS inside the browser) ----

    def _run_search(self, date_from, date_to, page_num):
        """Hit the transcript search endpoint for a single page."""
        body = {
            "text": "",
            "companies": [],
            "sectors": [],
            "eventTypes": ["Earnings Calls"],
            "fromDate": date_from,
            "toDate": date_to,
            "page": page_num,
            "pageSize": self._PER_PAGE,
            "typoTolerance": "false",
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw = self.driver.execute_script(_JS_RUN_SEARCH, self._jwt, body)
                data = json.loads(raw)

                if data.get("error") == "auth_expired":
                    self._relogin()
                    continue
                if data.get("error") == "rate_limited":
                    self.log.warning("Rate limited, sleeping 60s")
                    time.sleep(60)
                    continue
                if "error" in data:
                    self.log.warning("Search returned error: %s", data["error"])
                    time.sleep(5)
                    continue

                return data

            except Exception as exc:
                self.log.warning("Search failed (try %d): %s", attempt + 1, str(exc)[:80])
                time.sleep(3)

        return {"totalHits": 0, "hits": []}

    def _fetch_batch(self, transcript_ids):
        """Download multiple transcripts in one JS call."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw = self.driver.execute_script(_JS_FETCH_MANY, self._jwt, transcript_ids)
                results = json.loads(raw)

                if any(r.get("error") == "auth_expired" for r in results):
                    self._relogin()
                    continue
                if any(r.get("error") == "rate_limited" for r in results):
                    self.log.warning("Rate limited, sleeping 60s")
                    time.sleep(60)
                    continue

                return results

            except Exception as exc:
                self.log.warning("Batch fetch failed (try %d): %s", attempt + 1, str(exc)[:80])
                time.sleep(3)

        return [{"id": tid, "error": "max_retries"} for tid in transcript_ids]

    # ---- formatting ----

    @staticmethod
    def _build_text(hit_info, transcript_data):
        """
        Turn a transcript into a readable text file.
        Metadata header on top, then each speaker's text in order.
        """
        hdr = transcript_data.get("header", {})
        blocks = transcript_data.get("components", [])

        # metadata section
        lines = [
            f"Title: {hdr.get('transcriptTitle') or hit_info.get('transcriptTitle', '')}",
            f"Event Type: {hdr.get('eventType') or hit_info.get('eventType', '')}",
            f"Event Date: {hdr.get('eventDateTime') or hit_info.get('eventDateTime', '')}",
            f"Sector: {hit_info.get('sector', '')}",
            f"Industry: {hit_info.get('industry', '')}",
            f"Source: Koyfin (API)",
            f"Transcript ID: {hit_info.get('transcriptId', '')}",
            f"Scraped: {datetime.now().isoformat()}",
            "=" * 80,
        ]

        # speaker sections
        body = []
        for block in sorted(blocks, key=lambda b: b.get("order", 0)):
            name = block.get("speaker", "")
            role = block.get("role", "")
            text = block.get("text", "").strip()
            if not text:
                continue
            if name:
                body.append(f"\n{name}  [{role}]\n{text}")
            else:
                body.append(text)

        return "\n".join(lines) + "\n" + "\n\n".join(body) + "\n"

    # ---- adaptive date splitting ----

    def _split_date_range(self, start, end):
        """
        Recursively break up a date range if it has more than 1000 results
        (that's the most Koyfin will return).
        Returns a list of (start, end, count) tuples.
        """
        iso_start = f"{start}T00:00:00.000Z"
        iso_end = f"{end}T23:59:59.000Z"

        first_page = self._run_search(iso_start, iso_end, 0)
        total = first_page.get("totalHits", 0)

        if total == 0:
            return []

        if total <= self._HARD_CAP:
            return [(start, end, total)]

        # too many results — cut the range in half
        dt_start = datetime.strptime(start, "%Y-%m-%d")
        dt_end = datetime.strptime(end, "%Y-%m-%d")
        span = (dt_end - dt_start).days

        if span < 1:
            # single day with 1000+ calls, nothing we can do
            self.log.warning(
                "Day %s has %d results (> %d cap), will be truncated",
                start, total, self._HARD_CAP,
            )
            return [(start, end, total)]

        mid = dt_start + timedelta(days=span // 2)
        mid_str = mid.strftime("%Y-%m-%d")
        next_day = (mid + timedelta(days=1)).strftime("%Y-%m-%d")

        self.log.info(
            "Splitting %s..%s (%d hits) -> [%s..%s] + [%s..%s]",
            start, end, total,
            start, mid_str, next_day, end,
        )

        left = self._split_date_range(start, mid_str)
        right = self._split_date_range(next_day, end)
        return left + right

    # ---- main loop ----

    def scrape_range(self, start_date, end_date, dest, resume_file=None, batch_size=5):
        """
        Download all earnings call transcripts in a date range.

        Handles the 1000-result API cap by splitting ranges as needed.

        Args:
            start_date:  'YYYY-MM-DD'
            end_date:    'YYYY-MM-DD'
            dest:        output folder (transcripts go into year subfolders)
            resume_file: path to a JSON checkpoint to pick up where we left off
            batch_size:  how many transcripts to grab per JS call (1-10)

        Returns:
            Number of transcripts saved this run.
        """
        # load checkpoint if resuming
        done_ids = set()
        if resume_file and os.path.exists(resume_file):
            with open(resume_file, "r", encoding="utf-8") as fh:
                checkpoint = json.load(fh)
                done_ids = set(checkpoint.get("completed_ids", []))
            self.log.info("Resuming — %d transcripts already done", len(done_ids))

        # figure out sub-ranges
        chunks = self._split_date_range(start_date, end_date)
        expected_total = sum(n for _, _, n in chunks)

        self.log.info(
            "Range %s to %s: %d earnings calls across %d chunk(s)",
            start_date, end_date, expected_total, len(chunks),
        )

        if not chunks:
            return 0

        n_saved = 0
        n_skipped = 0
        n_errors = 0
        last_checkpoint = time.time()

        def write_checkpoint():
            nonlocal last_checkpoint
            if not resume_file:
                return
            os.makedirs(os.path.dirname(resume_file), exist_ok=True)
            with open(resume_file, "w", encoding="utf-8") as fh:
                json.dump({
                    "completed_ids": list(done_ids),
                    "saved_count": n_saved + n_skipped,
                    "last_updated": datetime.now().isoformat(),
                }, fh)
            last_checkpoint = time.time()

        for chunk_idx, (c_start, c_end, c_total) in enumerate(chunks):
            if halt_flag.is_set():
                break

            iso_from = f"{c_start}T00:00:00.000Z"
            iso_to = f"{c_end}T23:59:59.000Z"
            n_pages = min(
                (c_total + self._PER_PAGE - 1) // self._PER_PAGE,
                self._HARD_CAP // self._PER_PAGE,
            )

            for pg in range(n_pages):
                if halt_flag.is_set():
                    break

                page_data = self._run_search(iso_from, iso_to, pg)
                hits = page_data.get("hits", [])
                if not hits:
                    break

                # skip anything we already have
                pending = [h for h in hits if h["transcriptId"] not in done_ids]

                if not pending:
                    n_skipped += len(hits)
                    continue

                # process in batches
                for offset in range(0, len(pending), batch_size):
                    if halt_flag.is_set():
                        break

                    batch = pending[offset:offset + batch_size]
                    batch_ids = [h["transcriptId"] for h in batch]

                    transcripts = self._fetch_batch(batch_ids)

                    for hit, transcript in zip(batch, transcripts):
                        tid = hit["transcriptId"]
                        done_ids.add(tid)

                        if transcript.get("error"):
                            n_errors += 1
                            self.log.debug("Failed to get %d: %s", tid, transcript["error"])
                            continue

                        # figure out the year for the subfolder
                        event_dt = hit.get("eventDateTime", "")
                        try:
                            year = datetime.fromisoformat(
                                event_dt.replace("Z", "+00:00")
                            ).year
                        except Exception:
                            year = int(start_date[:4])

                        year_folder = os.path.join(dest, str(year))
                        os.makedirs(year_folder, exist_ok=True)

                        # build filename
                        title = hit.get("transcriptTitle", f"transcript_{tid}")
                        safe_name = sanitize_filename(title)
                        if len(safe_name) > 180:
                            safe_name = safe_name[:180]
                        fpath = os.path.join(year_folder, f"{safe_name}.txt")

                        if os.path.exists(fpath):
                            n_skipped += 1
                            continue

                        # format and write
                        content = self._build_text(hit, transcript)
                        if len(content) < 300:
                            n_errors += 1
                            continue

                        with open(fpath, "w", encoding="utf-8") as fh:
                            fh.write(content)
                        n_saved += 1

                        if n_saved % 100 == 0 or n_saved <= 5:
                            self.log.info(
                                "[chunk %d/%d pg %d] saved=%d skip=%d err=%d  %s",
                                chunk_idx + 1, len(chunks),
                                pg + 1, n_saved, n_skipped, n_errors,
                                title[:60],
                            )

                time.sleep(0.2)

            # checkpoint every couple minutes
            if time.time() - last_checkpoint > 120:
                write_checkpoint()

        write_checkpoint()
        self.log.info(
            "Finished: saved=%d, skipped=%d, errors=%d, expected=%d",
            n_saved, n_skipped, n_errors, expected_total,
        )
        return n_saved