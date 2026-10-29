"""Unit tests for scripts/export_replay.py integrity gates."""
from pathlib import Path
import pytest
from scripts.export_replay import (
    IntegrityError,
    verify_evidence_verbatim,
    verify_ledger,
    verify_trade_no_lookahead,
    verify_tz_aware,
    export_replay_payload,
)
from src.api.models import Evidence, RunSummary


def test_verify_tz_aware_passes_on_valid_utc():
    """Verify tz-aware timestamps succeed."""
    dt = verify_tz_aware("2026-10-29T12:00:00Z", "test_field")
    assert dt.tzinfo is not None

    dt2 = verify_tz_aware("2026-10-29T12:00:00+00:00", "test_field")
    assert dt2.tzinfo is not None


def test_verify_tz_aware_fails_on_naive():
    """Verify naive timestamp raises IntegrityError."""
    with pytest.raises(IntegrityError, match="not timezone-aware"):
        verify_tz_aware("2026-10-29T12:00:00", "naive_field")


def test_verify_no_lookahead_catches_violation():
    """Assert lookahead check fails loudly if trade entry is <= filing acceptance."""
    bad_trades = [
        {
            "ticker": "AAPL",
            "accepted_at": "2024-11-01T10:00:00Z",
            "execution_time": "2024-11-01T09:30:00Z",  # 30 mins before acceptance
        }
    ]
    with pytest.raises(IntegrityError, match="Lookahead violation"):
        verify_trade_no_lookahead(bad_trades)


def test_verify_evidence_verbatim_catches_divergence(tmp_path):
    """Assert evidence checker fails when quote is not a verbatim substring."""
    edgar_dir = tmp_path / "edgar"
    edgar_dir.mkdir()

    filing_file = edgar_dir / "TEST_0001.txt"
    filing_file.write_text("This is the exact real filing disclosure text from SEC EDGAR.", encoding="utf-8")

    # Divergent quote
    fake_evidence = [
        Evidence(
            claim_id="CLM-01",
            ticker="TEST",
            text="Claim about test",
            quote="This is a fabricated quote that does not exist in the filing.",
            accession="0001",
            filing_url="https://sec.gov",
            accepted_at="2024-11-01T12:00:00Z",
        )
    ]

    with pytest.raises(IntegrityError, match="NOT a verbatim substring"):
        verify_evidence_verbatim(fake_evidence, edgar_dir)


def test_verify_ledger_catches_tamper():
    """Assert corrupted ledger head hash raises IntegrityError."""
    blocks = ["block_1", "block_2"]
    with pytest.raises(IntegrityError, match="Ledger chain mismatch"):
        verify_ledger(blocks, "bad_head_hash_12345")


def test_export_replay_end_to_end(tmp_path):
    """Test full replay export produces valid RunSummary."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    models_config = repo_root / "config" / "models.yaml"
    edgar_dir = repo_root / "data" / "raw" / "edgar"
    out_dir = tmp_path / "replays"

    out_file = export_replay_payload(
        run_id="test-run-real-01",
        output_dir=out_dir,
        edgar_dir=edgar_dir,
        models_config_path=models_config,
        source="REAL",
    )

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    summary = RunSummary.model_validate_json(content)
    assert summary.id == "test-run-real-01"
    assert summary.source == "REAL"
    assert summary.backtest.n_events == 35
    assert summary.ledger.verified is True
