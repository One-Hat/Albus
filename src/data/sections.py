"""
SEC 10-K Section Parser: Item 1A (Risk Factors).
Robust extraction across HTML, plain text, and SEC EDGAR submission formats.
Ensures timezone-aware acceptance timestamps.
"""
from __future__ import annotations
import re
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

EASTERN = ZoneInfo("America/New_York")


class FilingSection(BaseModel):
    ticker: str
    accession_number: str
    form_type: str = "10-K"
    accepted_at: datetime
    section_name: str = "Item 1A"
    text: str
    word_count: int = Field(ge=0)


class Item1AParser:
    """
    Extracts Item 1A (Risk Factors) from 10-K filings.
    Filters out Table of Contents matches and strips enclosing metadata.
    """

    # Start pattern: ITEM 1A. RISK FACTORS (with various delimiters and punctuation)
    START_PATTERNS = [
        re.compile(r'(?i)\bitem\s+1a[\.\:\s\-\—]+risk\s+factors\b', re.MULTILINE),
        re.compile(r'(?i)\bitem\s+1a\b[^\n\r]{0,30}\brisk\s+factors\b', re.MULTILINE),
    ]

    # End pattern: ITEM 1B (Unresolved Staff Comments) or ITEM 2 (Properties)
    END_PATTERNS = [
        re.compile(r'(?i)\bitem\s+1b[\.\:\s\-\—]+unresolved\s+staff\s+comments\b', re.MULTILINE),
        re.compile(r'(?i)\bitem\s+2[\.\:\s\-\—]+properties\b', re.MULTILINE),
        re.compile(r'(?i)\bitem\s+1b\b', re.MULTILINE),
    ]

    def parse_acceptance_timestamp(self, accepted_at_str: str) -> datetime:
        """
        Parse acceptance timestamp into a timezone-aware datetime (Eastern Time).
        Supports formats:
        - YYYY-MM-DD HH:MM:SS
        - YYYYMMDDHHMMSS
        - ISO format strings
        """
        clean_str = accepted_at_str.strip()
        dt: Optional[datetime] = None

        if len(clean_str) == 14 and clean_str.isdigit():
            # SEC ACCEPTANCE-DATETIME: YYYYMMDDhhmmss
            dt = datetime.strptime(clean_str, "%Y%m%d%H%M%S")
        elif "T" in clean_str:
            dt = datetime.fromisoformat(clean_str)
        else:
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d"]:
                try:
                    dt = datetime.strptime(clean_str, fmt)
                    break
                except ValueError:
                    continue

        if dt is None:
            raise ValueError(f"Unable to parse acceptance timestamp: '{accepted_at_str}'")

        if dt.tzinfo is None:
            # SEC filings are stamped in US Eastern Time
            dt = dt.replace(tzinfo=EASTERN)

        return dt

    def clean_html(self, raw_html: str) -> str:
        """Strip HTML tags while preserving paragraph spacing."""
        if "<" not in raw_html or ">" not in raw_html:
            return raw_html

        soup = BeautifulSoup(raw_html, "html.parser")
        # Replace breaks and divs with newlines
        for br in soup.find_all(["br", "p", "div", "h1", "h2", "h3", "tr"]):
            br.append("\n")
        text = soup.get_text()
        return text

    def extract_item_1a(
        self,
        ticker: str,
        accession_number: str,
        raw_text: str,
        accepted_at_str: str,
    ) -> Optional[FilingSection]:
        """
        Locate and extract Item 1A section text from raw 10-K string.
        """
        accepted_at = self.parse_acceptance_timestamp(accepted_at_str)
        cleaned_text = self.clean_html(raw_text)

        # Normalize spaces
        normalized_lines = []
        for line in cleaned_text.splitlines():
            line_str = " ".join(line.split())
            if line_str:
                normalized_lines.append(line_str)
        full_text = "\n".join(normalized_lines)

        # Find all candidate start positions
        start_matches = []
        for pat in self.START_PATTERNS:
            for m in pat.finditer(full_text):
                start_matches.append(m)

        if not start_matches:
            return None

        # Sort matches by position in document
        start_matches.sort(key=lambda x: x.start())

        # Disambiguate Table of Contents vs Substantive section:
        # If there are multiple matches, the first one is often the TOC if followed quickly by Item 1B or small text.
        chosen_start = None
        chosen_end = None

        for idx, match in enumerate(start_matches):
            pos_after_start = match.end()
            remaining_text = full_text[pos_after_start:]

            # Look for end match after this start match
            best_end_pos = None
            for end_pat in self.END_PATTERNS:
                m_end = end_pat.search(remaining_text)
                if m_end:
                    if best_end_pos is None or m_end.start() < best_end_pos:
                        best_end_pos = m_end.start()

            if best_end_pos is not None:
                section_candidate = remaining_text[:best_end_pos].strip()
                # If section has more than 50 characters and not just dots/page numbers, choose it
                if len(section_candidate) > 50 and not section_candidate.endswith("..."):
                    chosen_start = pos_after_start
                    chosen_end = pos_after_start + best_end_pos
                    break
            else:
                # If no end marker found, take a reasonable chunk
                if len(remaining_text.strip()) > 50:
                    chosen_start = pos_after_start
                    chosen_end = len(full_text)
                    break

        if chosen_start is None or chosen_end is None:
            # Fallback to the last start match
            last_match = start_matches[-1]
            chosen_start = last_match.end()
            chosen_end = len(full_text)

        extracted_text = full_text[chosen_start:chosen_end].strip()
        word_count = len(extracted_text.split())

        return FilingSection(
            ticker=ticker,
            accession_number=accession_number,
            form_type="10-K",
            accepted_at=accepted_at,
            section_name="Item 1A",
            text=extracted_text,
            word_count=word_count,
        )
