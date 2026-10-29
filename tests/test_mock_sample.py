"""Validate that web/public/replays/mock-sample.json and index.json conform to contracts."""
import json
from pathlib import Path
import pytest
from src.api.models import RunIndexItem, RunSummary


def test_mock_sample_fixture_validates():
    """Verify that mock-sample.json exists and strictly validates against RunSummary."""
    repo_root = Path(__file__).resolve().parent.parent
    mock_file = repo_root / "web" / "public" / "replays" / "mock-sample.json"
    index_file = repo_root / "web" / "public" / "replays" / "index.json"

    assert mock_file.exists(), f"Mock fixture missing at {mock_file}"
    assert index_file.exists(), f"Index file missing at {index_file}"

    # Load and validate mock-sample.json
    with open(mock_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = RunSummary.model_validate(data)

    # Check required constraints from B0 spec:
    # 1. source must be 'MOCK'
    assert summary.source == "MOCK", f"Expected source 'MOCK', got {summary.source}"
    # 2. hypothesis must start with 'MOCK SAMPLE:'
    assert summary.hypothesis.startswith("MOCK SAMPLE:"), (
        f"Hypothesis must start with 'MOCK SAMPLE:', got {summary.hypothesis}"
    )
    # 3. 35 events
    assert summary.backtest.n_events == 35, (
        f"Expected 35 backtest events, got {summary.backtest.n_events}"
    )
    # 4. Include both calibration kinds
    kinds = {c.kind for c in summary.calibration}
    assert kinds == {"retrospective_clean", "live"}, (
        f"Expected both calibration kinds, got {kinds}"
    )
    # 5. One failed critic check
    failed_checks = [c for c in summary.verdict.checks if not c.passed]
    assert len(failed_checks) >= 1, "Must have at least one failed check for caveat states"

    # Validate index.json
    with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    assert isinstance(index_data, list), "index.json must be a list"
    assert len(index_data) >= 1, "index.json must contain at least one item"
    validated_index = [RunIndexItem.model_validate(item) for item in index_data]
    assert any(item.id == "mock-sample" for item in validated_index), (
        "mock-sample not found in index.json"
    )
