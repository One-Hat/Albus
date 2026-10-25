"""
SEC EDGAR Data Fetcher with Rate Limiting and User-Agent Compliance.
Adheres strictly to SEC Fair-Access rules (<= 10 requests/sec, declared User-Agent).
"""
from __future__ import annotations
import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import httpx

EASTERN = ZoneInfo("America/New_York")


class EDGARClient:
    """
    Client for SEC EDGAR submissions API and raw filing retrieval.
    Enforces rate limits and stores exact acceptance timestamps.
    """

    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{primary_doc}"

    def __init__(
        self,
        user_agent: Optional[str] = None,
        cache_dir: str | Path = "data/raw/edgar",
        requests_per_second: float = 8.0,
    ):
        raw_ua = (
            user_agent
            or os.environ.get("SEC_USER_AGENT")
            or os.environ.get("EDGAR_USER_AGENT")
            or "ALBUS-QuantResearchCopilot/1.0 (researcher@example.com)"
        )
        self.user_agent = raw_ua
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.min_request_interval_seconds = 1.0 / min(requests_per_second, 10.0)
        self._last_request_time = 0.0

    def _enforce_rate_limit(self) -> None:
        """Synchronous delay to respect SEC <= 10 req/s."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_request_interval_seconds:
            time.sleep(self.min_request_interval_seconds - elapsed)
        self._last_request_time = time.time()

    async def _async_enforce_rate_limit(self) -> None:
        """Asynchronous delay to respect SEC <= 10 req/s."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_request_interval_seconds:
            await asyncio.sleep(self.min_request_interval_seconds - elapsed)
        self._last_request_time = time.time()

    def get_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def parse_acceptance_datetime(self, sec_timestamp_str: str) -> datetime:
        """
        Parse SEC acceptance timestamp string into timezone-aware Eastern datetime.
        """
        # Formats: '2023-11-03T18:04:45.000Z' or '20231103180445'
        clean = sec_timestamp_str.strip()
        if len(clean) == 14 and clean.isdigit():
            dt = datetime.strptime(clean, "%Y%m%d%H%M%S")
            return dt.replace(tzinfo=EASTERN)
        if "T" in clean:
            dt = datetime.fromisoformat(clean.replace("Z", "+00:00"))
            return dt.astimezone(EASTERN)
        dt = datetime.strptime(clean, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=EASTERN)

    def fetch_filing_text_cached(
        self,
        ticker: str,
        accession_number: str,
    ) -> Optional[str]:
        """Check local disk cache for downloaded raw filing."""
        cache_file = self.cache_dir / f"{ticker}_{accession_number}.txt"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8", errors="ignore")
        return None

    def save_filing_text_cached(
        self,
        ticker: str,
        accession_number: str,
        content: str,
    ) -> None:
        """Save filing to local disk cache."""
        cache_file = self.cache_dir / f"{ticker}_{accession_number}.txt"
        cache_file.write_text(content, encoding="utf-8")

    def fetch_ticker_to_cik_map(self) -> Dict[str, str]:
        """Fetch mapping of uppercase ticker to 10-digit zero-padded CIK string."""
        cache_file = self.cache_dir / "company_tickers.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}
            except Exception:
                pass

        self._enforce_rate_limit()
        url = "https://www.sec.gov/files/company_tickers.json"
        with httpx.Client(timeout=30.0, headers=self.get_headers(), follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to fetch SEC ticker map: HTTP {resp.status_code}")
            cache_file.write_text(resp.text, encoding="utf-8")
            data = resp.json()
            return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}

    def fetch_company_submissions(self, cik: str) -> Dict[str, Any]:
        """Fetch recent submissions JSON for a company."""
        clean_cik = cik.zfill(10)
        cache_file = self.cache_dir / f"CIK{clean_cik}_submissions.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        self._enforce_rate_limit()
        url = self.SUBMISSIONS_URL.format(cik=clean_cik)
        with httpx.Client(timeout=30.0, headers=self.get_headers(), follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(f"Failed to fetch submissions for CIK {clean_cik}: HTTP {resp.status_code}")
            cache_file.write_text(resp.text, encoding="utf-8")
            return resp.json()

    def fetch_filing_document(
        self,
        ticker: str,
        cik: str,
        accession_number: str,
        primary_doc: str,
    ) -> str:
        """Fetch primary document text or HTML, using local cache if available."""
        cached = self.fetch_filing_text_cached(ticker, accession_number)
        if cached:
            return cached

        clean_cik = str(int(cik))  # SEC archives URL uses unpadded CIK
        nodash_acc = accession_number.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{clean_cik}/{nodash_acc}/{primary_doc}"

        self._enforce_rate_limit()
        with httpx.Client(timeout=60.0, headers=self.get_headers(), follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Failed to fetch filing document for {ticker} ({accession_number}): HTTP {resp.status_code}"
                )
            content = resp.text
            self.save_filing_text_cached(ticker, accession_number, content)
            return content
