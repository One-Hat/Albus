"""
Question generator turning each FilingDiff into a falsifiable forecasting Question.
Enforces post-knowledge-cutoff check and separates retrospective_clean vs live forecasts.
"""
from __future__ import annotations
import hashlib
from datetime import datetime, timedelta
from typing import Literal, Optional
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field, field_validator

from extract.quote_check import FilingDiff

EASTERN = ZoneInfo("America/New_York")
DEFAULT_KNOWLEDGE_CUTOFF = datetime(2023, 1, 1, 0, 0, tzinfo=EASTERN)


class Question(BaseModel):
    question_id: str
    ticker: str
    event_def: str  # e.g. 60-trading-day excess return vs sector ETF is negative
    horizon_days: int = Field(ge=1)
    resolves_at: datetime
    kind: Literal["retrospective_clean", "live"]

    @field_validator("resolves_at")
    @classmethod
    def check_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Question.resolves_at must be timezone-aware.")
        return v


class QuestionGenerator:
    """
    Converts FilingDiff records into standardized binary forecasting questions.
    Rejects filings accepted before the model knowledge cutoff (Design Decision 3a).
    """

    def __init__(self, knowledge_cutoff: datetime = DEFAULT_KNOWLEDGE_CUTOFF):
        if knowledge_cutoff.tzinfo is None:
            knowledge_cutoff = knowledge_cutoff.replace(tzinfo=EASTERN)
        self.knowledge_cutoff = knowledge_cutoff

    def from_filing_diff(
        self,
        diff: FilingDiff,
        horizon_days: int = 60,
        event_def: Optional[str] = None,
        as_of: Optional[datetime] = None,
    ) -> Question:
        accepted_et = diff.accepted_at.astimezone(EASTERN)
        if accepted_et < self.knowledge_cutoff:
            raise ValueError(
                f"Filing for {diff.ticker} at {accepted_et.isoformat()} is before model "
                f"knowledge cutoff ({self.knowledge_cutoff.isoformat()}). Rejected to prevent leakage."
            )

        reference_now = (as_of or datetime.now(EASTERN)).astimezone(EASTERN)
        # Convert trading days to approximate calendar days (trading_days * 7 / 5 + holidays buffer)
        calendar_days = int(horizon_days * 7 / 5) + 3
        resolves_at = (accepted_et + timedelta(days=calendar_days)).replace(
            hour=16, minute=0, second=0, microsecond=0
        )

        kind: Literal["retrospective_clean", "live"] = (
            "retrospective_clean" if resolves_at <= reference_now else "live"
        )
        definition = (
            event_def
            or f"{horizon_days}-trading-day excess return vs sector ETF is negative"
        )

        raw_id = f"{diff.ticker}|{accepted_et.isoformat()}|{horizon_days}|{definition}"
        qid = "q_" + hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]

        return Question(
            question_id=qid,
            ticker=diff.ticker,
            event_def=definition,
            horizon_days=horizon_days,
            resolves_at=resolves_at,
            kind=kind,
        )
