"""
Unit tests for Phase 3: Filing-diff extractor (Nano) and verbatim quote verification.
Following AGENTS.md Rule 2: Tests written first.
"""
import json
import shutil
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo
import pytest
from pydantic import ValidationError

from llm.mock import MockLLM
from llm.cache import DiskCache
from llm.client import LLMClient
from data.sections import FilingSection
from extract.quote_check import Claim, FilingDiff, is_verbatim_quote, verify_and_filter_diff
from extract.diff_extractor import DiffExtractor

EASTERN = ZoneInfo("America/New_York")


@pytest.fixture
def temp_cache_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_extract_cache_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_data_contracts_claim_and_filing_diff():
    """Verify Claim and FilingDiff match Section 5 data contracts."""
    claim = Claim(
        text="Increased litigation risk in patent disputes",
        quote="We face three new patent infringement lawsuits in Texas.",
        source_id="AAPL_0000320193-23-000106",
    )
    assert claim.text == "Increased litigation risk in patent disputes"
    assert claim.quote == "We face three new patent infringement lawsuits in Texas."
    assert claim.source_id == "AAPL_0000320193-23-000106"

    dt = datetime(2023, 11, 3, 18, 4, 45, tzinfo=EASTERN)
    diff = FilingDiff(
        ticker="AAPL",
        accepted_at=dt,
        new_risks=[claim],
        removed_risks=[],
        tone_shift=-0.35,
    )
    assert diff.ticker == "AAPL"
    assert diff.accepted_at.tzinfo is not None
    assert len(diff.new_risks) == 1
    assert diff.tone_shift == -0.35

    # Tone shift outside -1..1 should fail validation
    with pytest.raises(ValidationError):
        FilingDiff(
            ticker="AAPL",
            accepted_at=dt,
            new_risks=[],
            removed_risks=[],
            tone_shift=1.5,
        )

    # Naive datetime should fail validation
    with pytest.raises(ValidationError):
        FilingDiff(
            ticker="AAPL",
            accepted_at=datetime(2023, 11, 3, 18, 4, 45),
            new_risks=[],
            removed_risks=[],
            tone_shift=0.0,
        )


def test_planted_fake_quote_is_rejected():
    """
    Gate requirement: quote_check.py rejects any claim whose quote is not a verbatim substring,
    and a planted fake quote is rejected.
    """
    source_t = (
        "Our business faces intense competition. In 2023, we became subject to several class-action "
        "lawsuits regarding product warranty disclosures. Supply chain bottlenecks have also persisted."
    )
    source_t_minus_1 = (
        "Our business faces intense competition. Temporary store closures due to pandemic lockdowns "
        "affected retail sales in 2022."
    )

    valid_new_claim = Claim(
        text="New class-action lawsuits on warranties",
        quote="we became subject to several class-action lawsuits regarding product warranty disclosures.",
        source_id="TEST_2023",
    )
    planted_fake_claim = Claim(
        text="Hallucinated antitrust fine",
        quote="we were fined $5 billion by antitrust regulators in 2023.",
        source_id="TEST_2023",
    )
    valid_removed_claim = Claim(
        text="Removed pandemic store closure risk",
        quote="Temporary store closures due to pandemic lockdowns affected retail sales in 2022.",
        source_id="TEST_2022",
    )

    assert is_verbatim_quote(valid_new_claim.quote, source_t) is True
    assert is_verbatim_quote(planted_fake_claim.quote, source_t) is False
    assert is_verbatim_quote(valid_removed_claim.quote, source_t_minus_1) is True

    raw_diff = FilingDiff(
        ticker="TEST",
        accepted_at=datetime(2023, 11, 3, 18, 0, tzinfo=EASTERN),
        new_risks=[valid_new_claim, planted_fake_claim],
        removed_risks=[valid_removed_claim],
        tone_shift=-0.25,
    )

    cleaned_diff, stats = verify_and_filter_diff(raw_diff, source_t, source_t_minus_1)
    assert len(cleaned_diff.new_risks) == 1
    assert cleaned_diff.new_risks[0].quote == valid_new_claim.quote
    assert len(cleaned_diff.removed_risks) == 1
    assert stats["rejected_count"] == 1
    assert stats["kept_count"] == 2


@pytest.mark.asyncio
async def test_diff_extractor_repair_loop(temp_cache_dir):
    """
    Verify DiffExtractor compares year t vs t-1 using Nano and runs validate-and-repair
    when an initial response contains an invalid/non-verbatim quote.
    """
    sec_t_minus_1 = FilingSection(
        ticker="NVDA",
        accession_number="0001045810-23-000017",
        accepted_at=datetime(2023, 3, 3, 16, 30, tzinfo=EASTERN),
        text="We rely on third-party foundries. Legacy gaming GPU inventory excess may hurt margins.",
        word_count=14,
    )
    sec_t = FilingSection(
        ticker="NVDA",
        accession_number="0001045810-24-000029",
        accepted_at=datetime(2024, 2, 21, 16, 42, tzinfo=EASTERN),
        text="We rely on third-party foundries. Export control restrictions on AI chips may materially impact revenue.",
        word_count=15,
    )

    mock = MockLLM()
    # Attempt 1: emits a non-verbatim paraphrased quote -> triggers repair
    # Attempt 2: emits exact verbatim substring -> passes
    bad_payload = {
        "new_risks": [
            {
                "text": "Export controls on AI chips",
                "quote": "Export controls on AI chips might hurt revenue.",  # NOT verbatim
                "source_id": "0001045810-24-000029",
            }
        ],
        "removed_risks": [],
        "tone_shift": -0.4,
    }
    good_payload = {
        "new_risks": [
            {
                "text": "Export controls on AI chips",
                "quote": "Export control restrictions on AI chips may materially impact revenue.",
                "source_id": "0001045810-24-000029",
            }
        ],
        "removed_risks": [
            {
                "text": "Legacy gaming GPU inventory excess removed",
                "quote": "Legacy gaming GPU inventory excess may hurt margins.",
                "source_id": "0001045810-23-000017",
            }
        ],
        "tone_shift": -0.4,
    }

    mock.register_sequence(
        role_or_model="nano",
        responses=[
            {"content": json.dumps(bad_payload)},
            {"content": json.dumps(good_payload)},
        ],
    )

    cache = DiskCache(cache_dir=temp_cache_dir)
    client = LLMClient(cache=cache, mock_provider=mock)
    extractor = DiffExtractor(llm_client=client)

    diff = await extractor.extract_diff(sec_t, sec_t_minus_1)
    assert diff.ticker == "NVDA"
    assert len(diff.new_risks) == 1
    assert diff.new_risks[0].quote == "Export control restrictions on AI chips may materially impact revenue."
    assert len(diff.removed_risks) == 1
    assert diff.removed_risks[0].quote == "Legacy gaming GPU inventory excess may hurt margins."


@pytest.mark.asyncio
async def test_gate_50_filings_quote_verification(temp_cache_dir):
    """
    Gate requirement: On 50 filings, at least 98% of kept quotes verify,
    and a planted fake quote is rejected.
    """
    mock = MockLLM()
    cache = DiskCache(cache_dir=temp_cache_dir)
    client = LLMClient(cache=cache, mock_provider=mock)
    extractor = DiffExtractor(llm_client=client)

    pairs = []
    responses = []

    for i in range(50):
        ticker = f"TICK{i:02d}"
        text_prev = (
            f"Company {ticker} operates globally. "
            f"In year {2022}, raw material inflation of {i + 5}% pressured operating margins. "
            f"Standard credit facility covenants apply."
        )
        text_curr = (
            f"Company {ticker} operates globally. "
            f"In year {2023}, a new regulatory investigation into data privacy in region {i} "
            f"could result in material fines and compliance costs. "
            f"Standard credit facility covenants apply."
        )

        sec_prev = FilingSection(
            ticker=ticker,
            accession_number=f"000000-{22}-{i:04d}",
            accepted_at=datetime(2023, 2, 15, 16, 30, tzinfo=EASTERN),
            text=text_prev,
            word_count=len(text_prev.split()),
        )
        sec_curr = FilingSection(
            ticker=ticker,
            accession_number=f"000000-{23}-{i:04d}",
            accepted_at=datetime(2024, 2, 15, 16, 30, tzinfo=EASTERN),
            text=text_curr,
            word_count=len(text_curr.split()),
        )
        pairs.append((sec_curr, sec_prev))

        new_claims = [
            {
                "text": f"New regulatory privacy investigation in region {i}",
                "quote": f"a new regulatory investigation into data privacy in region {i} could result in material fines and compliance costs.",
                "source_id": sec_curr.accession_number,
            }
        ]
        # Plant a fake quote on filing #10 to ensure it gets rejected even if repair exhausted
        if i == 10:
            new_claims.append(
                {
                    "text": "Planted fake bankruptcy claim",
                    "quote": "THIS QUOTE DOES NOT EXIST IN ANY FILING AT ALL.",
                    "source_id": sec_curr.accession_number,
                }
            )

        removed_claims = [
            {
                "text": "Removed raw material inflation risk",
                "quote": f"In year 2022, raw material inflation of {i + 5}% pressured operating margins.",
                "source_id": sec_prev.accession_number,
            }
        ]

        payload = {
            "new_risks": new_claims,
            "removed_risks": removed_claims,
            "tone_shift": -0.2,
        }
        # If i == 10, repair loop will try up to max_repairs times; register for those attempts
        if i == 10:
            for _ in range(3):
                responses.append({"content": json.dumps(payload)})
        else:
            responses.append({"content": json.dumps(payload)})

    mock.register_sequence("nano", responses)

    total_kept_quotes = 0
    verified_kept_quotes = 0
    planted_fake_survived = False

    for idx, (sec_curr, sec_prev) in enumerate(pairs):
        diff = await extractor.extract_diff(sec_curr, sec_prev, max_repairs=2)
        for claim in diff.new_risks:
            total_kept_quotes += 1
            if "THIS QUOTE DOES NOT EXIST" in claim.quote:
                planted_fake_survived = True
            if is_verbatim_quote(claim.quote, sec_curr.text):
                verified_kept_quotes += 1

        for claim in diff.removed_risks:
            total_kept_quotes += 1
            if is_verbatim_quote(claim.quote, sec_prev.text):
                verified_kept_quotes += 1

    assert not planted_fake_survived, "Planted fake quote must be rejected!"
    assert total_kept_quotes >= 100
    verification_rate = verified_kept_quotes / total_kept_quotes
    assert verification_rate >= 0.98, f"Expected >= 98% quote verification rate, got {verification_rate:.2%}"
