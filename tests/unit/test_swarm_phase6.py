"""
Unit and Gate tests for Phase 6: Forecasting swarm (questions.py, agents.py, ledger.py).
Following AGENTS.md Rule 2: Tests written first.
"""
import json
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pytest
from pydantic import ValidationError

from llm.mock import MockLLM
from llm.cache import DiskCache
from llm.client import LLMClient
from extract.quote_check import Claim, FilingDiff
from swarm.questions import Question, QuestionGenerator
from swarm.agents import (
    NANO_PERSONAS,
    SUPER_PERSONAS,
    ALL_PERSONAS,
    SwarmAgentOutput,
    SwarmRunner,
)
from swarm.ledger import Forecast, ForecastLedger, LedgerTamperError

EASTERN = ZoneInfo("America/New_York")


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="test_swarm_p6_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_question_generator_and_contracts():
    """
    Verify Question contract and QuestionGenerator (retrospective_clean vs live, plus cutoff guard).
    """
    cutoff = datetime(2023, 1, 1, 0, 0, tzinfo=EASTERN)
    now_dt = datetime(2024, 6, 1, 12, 0, tzinfo=EASTERN)
    gen = QuestionGenerator(knowledge_cutoff=cutoff)

    diff_retro = FilingDiff(
        ticker="AAPL",
        accepted_at=datetime(2023, 11, 3, 18, 4, tzinfo=EASTERN),
        new_risks=[
            Claim(
                text="New lawsuit risk",
                quote="We face several new patent lawsuits.",
                source_id="AAPL_2023",
            )
        ],
        removed_risks=[],
        tone_shift=-0.35,
    )
    q_retro: Question = gen.from_filing_diff(
        diff_retro,
        horizon_days=60,
        as_of=now_dt,
    )
    assert q_retro.ticker == "AAPL"
    assert q_retro.horizon_days == 60
    assert q_retro.kind == "retrospective_clean"
    assert q_retro.resolves_at > diff_retro.accepted_at

    # Live question where resolves_at is in the future relative to as_of
    diff_live = FilingDiff(
        ticker="NVDA",
        accepted_at=datetime(2024, 5, 20, 16, 30, tzinfo=EASTERN),
        new_risks=[],
        removed_risks=[],
        tone_shift=-0.1,
    )
    q_live = gen.from_filing_diff(diff_live, horizon_days=60, as_of=now_dt)
    assert q_live.kind == "live"

    # Pre-cutoff filing must be rejected to prevent model memory leakage (Design Decision 3a)
    diff_old = FilingDiff(
        ticker="MSFT",
        accepted_at=datetime(2021, 7, 29, 16, 15, tzinfo=EASTERN),
        new_risks=[],
        removed_risks=[],
        tone_shift=0.0,
    )
    with pytest.raises(ValueError, match="knowledge cutoff"):
        gen.from_filing_diff(diff_old, horizon_days=60, as_of=now_dt)


def test_personas_definition():
    """
    Verify agents.py defines 8 personas on Nano and 2 to 3 on Super.
    """
    assert len(NANO_PERSONAS) == 8, f"Expected 8 Nano personas, got {len(NANO_PERSONAS)}"
    assert 2 <= len(SUPER_PERSONAS) <= 3, f"Expected 2..3 Super personas, got {len(SUPER_PERSONAS)}"
    assert all(p.role == "nano" for p in NANO_PERSONAS)
    assert all(p.role == "super" for p in SUPER_PERSONAS)
    assert len(ALL_PERSONAS) == len(NANO_PERSONAS) + len(SUPER_PERSONAS)


def test_gate_forecasts_outside_bounds_rejected(temp_dir):
    """
    Gate requirement: Forecasts outside 0.02 to 0.98 are rejected.
    """
    dt = datetime(2024, 2, 22, 10, 0, tzinfo=EASTERN)

    # Too low (< 0.02)
    with pytest.raises(ValidationError):
        Forecast(
            forecast_id="f1",
            question_id="q1",
            agent_id="nano_skeptic",
            model="nvidia/nemotron-3-nano-30b-a3b",
            p_yes=0.01,
            rationale="Too low probability",
            evidence_ids=["AAPL_2023"],
            made_at=dt,
            prev_hash="GENESIS",
            hash="dummy",
        )

    # Too high (> 0.98)
    with pytest.raises(ValidationError):
        Forecast(
            forecast_id="f2",
            question_id="q1",
            agent_id="nano_skeptic",
            model="nvidia/nemotron-3-nano-30b-a3b",
            p_yes=0.995,
            rationale="Too high probability",
            evidence_ids=["AAPL_2023"],
            made_at=dt,
            prev_hash="GENESIS",
            hash="dummy",
        )

    # Ledger.append_forecast also rejects out-of-bounds p_yes
    ledger = ForecastLedger(ledger_path=temp_dir / "ledger.jsonl")
    with pytest.raises(ValidationError):
        ledger.append_forecast(
            question_id="q1",
            agent_id="nano_skeptic",
            model="nvidia/nemotron-3-nano-30b-a3b",
            p_yes=0.005,
            rationale="Invalid low",
            evidence_ids=[],
            made_at=dt,
        )


@pytest.mark.asyncio
async def test_gate_ledger_verifies_end_to_end_and_tamper_breaks_verification(temp_dir):
    """
    Gate requirement:
    - Each agent returns p_yes, a rationale and evidence IDs.
    - ledger.py appends hash-chained records.
    - The ledger verifies end to end, and editing any past record breaks verification.
    """
    mock = MockLLM()
    # Register responses for Nano (8 calls) and Super (3 calls)
    nano_seq = [
        {
            "content": json.dumps({
                "p_yes": round(0.30 + i * 0.05, 2),
                "rationale": f"Nano persona {i} analysis of new litigation risks.",
                "evidence_ids": ["AAPL_0000320193-23-000106"],
            })
        }
        for i in range(8)
    ]
    super_seq = [
        {
            "content": json.dumps({
                "p_yes": round(0.55 + j * 0.05, 2),
                "rationale": f"Super persona {j} synthesis of historical base rate and filing diff.",
                "evidence_ids": ["AAPL_0000320193-23-000106"],
            })
        }
        for j in range(len(SUPER_PERSONAS))
    ]
    mock.register_sequence("nano", nano_seq)
    mock.register_sequence("super", super_seq)

    cache = DiskCache(cache_dir=temp_dir / "cache")
    client = LLMClient(cache=cache, mock_provider=mock)

    ledger_file = temp_dir / "forecasts.jsonl"
    ledger = ForecastLedger(ledger_path=ledger_file)
    runner = SwarmRunner(llm_client=client, ledger=ledger)

    diff = FilingDiff(
        ticker="AAPL",
        accepted_at=datetime(2023, 11, 3, 18, 4, tzinfo=EASTERN),
        new_risks=[
            Claim(
                text="Patent litigation risk added",
                quote="We face several new patent lawsuits in 2023.",
                source_id="AAPL_0000320193-23-000106",
            )
        ],
        removed_risks=[],
        tone_shift=-0.35,
    )
    gen = QuestionGenerator(knowledge_cutoff=datetime(2023, 1, 1, tzinfo=EASTERN))
    question = gen.from_filing_diff(
        diff,
        horizon_days=60,
        as_of=datetime(2024, 6, 1, tzinfo=EASTERN),
    )

    forecasts = await runner.forecast_question(
        question=question,
        diff=diff,
        made_at=datetime(2023, 11, 4, 9, 0, tzinfo=EASTERN),
    )

    assert len(forecasts) == len(ALL_PERSONAS)
    for f in forecasts:
        assert 0.02 <= f.p_yes <= 0.98
        assert len(f.rationale) > 0
        assert "AAPL_0000320193-23-000106" in f.evidence_ids

    # Verify hash-chained ledger end-to-end
    assert ledger.verify() is True
    records = ledger.read_all()
    assert len(records) == len(ALL_PERSONAS)
    assert records[0].prev_hash == "0" * 64
    for idx in range(1, len(records)):
        assert records[idx].prev_hash == records[idx - 1].hash

    # Now tamper with a past record (e.g. change p_yes on record #2) and verify it breaks verification!
    lines = ledger_file.read_text(encoding="utf-8").splitlines()
    tampered_obj = json.loads(lines[2])
    tampered_obj["p_yes"] = 0.88  # Modified without recomputing chain hashes
    lines[2] = json.dumps(tampered_obj)
    ledger_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(LedgerTamperError):
        ledger.verify(raise_on_error=True)
    assert ledger.verify(raise_on_error=False) is False
