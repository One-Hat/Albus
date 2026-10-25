"""
Data contracts and verbatim quote verification for SEC filing diffs.
AGENTS.md Rule 6: Every LLM output is validated by pydantic. Quotes must be verbatim substrings.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, Field, field_validator


class Claim(BaseModel):
    text: str
    quote: str  # verbatim substring of the source section
    source_id: str


class FilingDiff(BaseModel):
    ticker: str
    accepted_at: datetime  # SEC acceptance time, tz-aware
    new_risks: List[Claim]
    removed_risks: List[Claim]
    tone_shift: float = Field(ge=-1.0, le=1.0)

    @field_validator("accepted_at")
    @classmethod
    def check_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("FilingDiff.accepted_at must be timezone-aware.")
        return v


def is_verbatim_quote(quote: str, source_text: str) -> bool:
    """
    Check if `quote` is a verbatim substring of `source_text`.
    Rejects empty quotes or whitespace-only quotes.
    """
    clean_quote = quote.strip()
    if not clean_quote:
        return False

    if clean_quote in source_text:
        return True

    # Also allow matching when multi-line wrapping in source_text is normalized to single spaces
    norm_quote = " ".join(clean_quote.split())
    norm_source = " ".join(source_text.split())
    return norm_quote in norm_source


def find_unverified_quotes(
    new_risks: List[Claim],
    removed_risks: List[Claim],
    source_t: str,
    source_t_minus_1: str,
) -> List[str]:
    """Return a list of error descriptions for any claims whose quotes are not verbatim substrings."""
    errors: List[str] = []
    for idx, claim in enumerate(new_risks):
        if not is_verbatim_quote(claim.quote, source_t):
            errors.append(
                f"new_risks[{idx}] quote is NOT a verbatim substring of year t filing: '{claim.quote}'"
            )
    for idx, claim in enumerate(removed_risks):
        if not is_verbatim_quote(claim.quote, source_t_minus_1):
            errors.append(
                f"removed_risks[{idx}] quote is NOT a verbatim substring of year t-1 filing: '{claim.quote}'"
            )
    return errors


def verify_and_filter_diff(
    diff: FilingDiff,
    source_t: str,
    source_t_minus_1: str,
) -> Tuple[FilingDiff, Dict[str, Any]]:
    """
    Reject any claim whose quote is not a verbatim substring of its respective source filing.
    Returns (filtered_diff, stats).
    """
    kept_new: List[Claim] = []
    rejected_new: List[Claim] = []
    for claim in diff.new_risks:
        if is_verbatim_quote(claim.quote, source_t):
            kept_new.append(claim)
        else:
            rejected_new.append(claim)

    kept_removed: List[Claim] = []
    rejected_removed: List[Claim] = []
    for claim in diff.removed_risks:
        if is_verbatim_quote(claim.quote, source_t_minus_1):
            kept_removed.append(claim)
        else:
            rejected_removed.append(claim)

    kept_count = len(kept_new) + len(kept_removed)
    rejected_count = len(rejected_new) + len(rejected_removed)
    total_count = kept_count + rejected_count

    filtered = FilingDiff(
        ticker=diff.ticker,
        accepted_at=diff.accepted_at,
        new_risks=kept_new,
        removed_risks=kept_removed,
        tone_shift=diff.tone_shift,
    )
    stats = {
        "kept_count": kept_count,
        "rejected_count": rejected_count,
        "total_count": total_count,
        "verification_rate": (kept_count / total_count) if total_count > 0 else 1.0,
    }
    return filtered, stats
