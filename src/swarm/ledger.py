"""
Append-only, hash-chained Forecast Ledger.
Guarantees tamper-evident storage of predictions before outcomes exist,
and rejects any forecast with p_yes outside [0.02, 0.98].
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field, field_validator

EASTERN = ZoneInfo("America/New_York")
GENESIS_HASH = "0" * 64


class LedgerTamperError(ValueError):
    """Raised when hash chain verification detects tampering or corruption in the forecast ledger."""


class Forecast(BaseModel):
    forecast_id: str
    question_id: str
    agent_id: str
    model: str
    p_yes: float = Field(ge=0.02, le=0.98)  # Gate: forecasts outside 0.02..0.98 are rejected
    rationale: str
    evidence_ids: List[str]
    made_at: datetime
    prev_hash: str
    hash: str  # sha256 over this record plus prev_hash

    @field_validator("made_at")
    @classmethod
    def check_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Forecast.made_at must be timezone-aware.")
        return v


def compute_record_hash(
    forecast_id: str,
    question_id: str,
    agent_id: str,
    model: str,
    p_yes: float,
    rationale: str,
    evidence_ids: List[str],
    made_at: datetime,
    prev_hash: str,
) -> str:
    """Compute canonical SHA-256 hash over the forecast record fields plus prev_hash."""
    canonical = {
        "forecast_id": forecast_id,
        "question_id": question_id,
        "agent_id": agent_id,
        "model": model,
        "p_yes": round(float(p_yes), 6),
        "rationale": rationale,
        "evidence_ids": list(evidence_ids),
        "made_at": made_at.isoformat(),
        "prev_hash": prev_hash,
    }
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ForecastLedger:
    """
    Append-only JSONL ledger chaining every forecast via SHA-256 hashes.
    """

    def __init__(self, ledger_path: str | Path = "data/ledger/forecasts.jsonl"):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def read_all(self) -> List[Forecast]:
        """Read and parse all Forecast records from disk."""
        if not self.ledger_path.exists():
            return []
        records: List[Forecast] = []
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                data = json.loads(stripped)
                records.append(Forecast.model_validate(data))
        return records

    def last_hash(self) -> str:
        """Return the hash of the most recent record, or GENESIS_HASH if empty."""
        records = self.read_all()
        if not records:
            return GENESIS_HASH
        return records[-1].hash

    def append_forecast(
        self,
        question_id: str,
        agent_id: str,
        model: str,
        p_yes: float,
        rationale: str,
        evidence_ids: List[str],
        made_at: Optional[datetime] = None,
        forecast_id: Optional[str] = None,
    ) -> Forecast:
        """
        Validate, hash-chain, and append a new Forecast record to the ledger.
        """
        timestamp = (made_at or datetime.now(EASTERN)).astimezone(EASTERN)
        prev = self.last_hash()

        fid = forecast_id or (
            "fc_"
            + hashlib.sha256(
                f"{question_id}|{agent_id}|{timestamp.isoformat()}|{prev}".encode("utf-8")
            ).hexdigest()[:16]
        )

        # Pre-validate bounds before computing hash (raises ValidationError if outside 0.02..0.98)
        record_hash = compute_record_hash(
            forecast_id=fid,
            question_id=question_id,
            agent_id=agent_id,
            model=model,
            p_yes=p_yes,
            rationale=rationale,
            evidence_ids=evidence_ids,
            made_at=timestamp,
            prev_hash=prev,
        )

        forecast = Forecast(
            forecast_id=fid,
            question_id=question_id,
            agent_id=agent_id,
            model=model,
            p_yes=round(float(p_yes), 6),
            rationale=rationale,
            evidence_ids=list(evidence_ids),
            made_at=timestamp,
            prev_hash=prev,
            hash=record_hash,
        )

        payload = forecast.model_dump(mode="json")
        # Keep ISO string format consistent with compute_record_hash
        payload["made_at"] = timestamp.isoformat()

        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

        return forecast

    def verify(self, raise_on_error: bool = False) -> bool:
        """
        Verify the entire hash chain from genesis to the latest record.
        Any edit to a past record breaks verification.
        """
        if not self.ledger_path.exists():
            return True

        try:
            records = self.read_all()
        except Exception as err:
            if raise_on_error:
                raise LedgerTamperError(f"Ledger failed schema validation: {err}") from err
            return False

        expected_prev = GENESIS_HASH
        for idx, rec in enumerate(records):
            if rec.prev_hash != expected_prev:
                msg = (
                    f"Chain broken at record {idx} ({rec.forecast_id}): "
                    f"expected prev_hash={expected_prev}, got {rec.prev_hash}"
                )
                if raise_on_error:
                    raise LedgerTamperError(msg)
                return False

            recomputed = compute_record_hash(
                forecast_id=rec.forecast_id,
                question_id=rec.question_id,
                agent_id=rec.agent_id,
                model=rec.model,
                p_yes=rec.p_yes,
                rationale=rec.rationale,
                evidence_ids=rec.evidence_ids,
                made_at=rec.made_at,
                prev_hash=rec.prev_hash,
            )
            if recomputed != rec.hash:
                msg = (
                    f"Tamper detected at record {idx} ({rec.forecast_id}): "
                    f"recomputed hash {recomputed} != stored hash {rec.hash}"
                )
                if raise_on_error:
                    raise LedgerTamperError(msg)
                return False

            expected_prev = rec.hash

        return True
