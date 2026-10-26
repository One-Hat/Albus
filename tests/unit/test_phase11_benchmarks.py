"""
Unit tests for Phase 11: Benchmark Harness (scripts/run_benchmarks.py).
Tests that the harness strictly guards against mock execution and missing keys,
and that the benchmark metrics calculation and schemas adhere to specifications.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from scripts.run_benchmarks import guard_check, calculate_bootstrap_brier_ci, BenchmarkReportSchema


def test_guard_check_exits_without_api_key(monkeypatch):
    """Verify that guard_check() immediately terminates with code 1 if no live API key is set."""
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        guard_check()
    assert exc.value.code == 1


def test_guard_check_subprocess_exit_nonzero():
    """Verify running scripts/run_benchmarks.py via subprocess exits non-zero and prints clear error."""
    env = os.environ.copy()
    env["NEBIUS_API_KEY"] = ""
    env["GROQ_API_KEY"] = ""
    env["OPENROUTER_API_KEY"] = ""

    res = subprocess.run(
        [sys.executable, "-c", "import os; os.environ['GROQ_API_KEY']=''; os.environ['NEBIUS_API_KEY']=''; from scripts.run_benchmarks import guard_check; guard_check()"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode != 0, f"Expected non-zero exit code, got {res.returncode}"
    assert "FATAL GUARD ERROR" in res.stderr


def test_guard_check_rejects_mock_llm(monkeypatch):
    """Verify that guard_check() aborts if MockLLM is present in sys.modules."""
    monkeypatch.setenv("NEBIUS_API_KEY", "dummy-key-for-test")
    # Simulate MockLLM module being imported
    sys.modules["src.llm.mock"] = object()  # type: ignore
    try:
        with pytest.raises(SystemExit) as exc:
            guard_check()
        assert exc.value.code == 1
    finally:
        sys.modules.pop("src.llm.mock", None)


def test_bootstrap_brier_ci_computation():
    """Verify bootstrap confidence interval computation for Brier scores."""
    # Synthetic forecast probabilities and binary outcomes
    forecasts = np.array([0.8, 0.7, 0.9, 0.6, 0.85, 0.2, 0.15, 0.3, 0.25, 0.1])
    outcomes = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    mean_brier, (low, high) = calculate_bootstrap_brier_ci(
        forecasts=forecasts,
        outcomes=outcomes,
        n_bootstraps=500,
        ci=0.95,
        seed=42,
    )

    assert 0.0 <= mean_brier <= 0.2
    assert low <= mean_brier <= high
    assert 0.0 <= low < high <= 1.0


def test_benchmark_schema_validation():
    """Verify that BenchmarkReportSchema validates required fields for all 4 benchmark pillars."""
    sample_report = {
        "timestamp_utc": "2026-10-26T12:00:00Z",
        "status": "completed",
        "models": {
            "nano": "nvidia/nemotron-3-nano-30b-a3b",
            "super": "nvidia/nemotron-3-super-120b-a12b",
            "ultra": "nvidia/Nemotron-3-Ultra-550b-a55b",
        },
        "extraction": {
            "nano": {
                "filings_per_min": 142.5,
                "cost_per_1k_filings": 0.42,
                "quote_verify_rate": 0.992,
            },
            "ultra_only": {
                "filings_per_min": 11.8,
                "cost_per_1k_filings": 12.80,
                "quote_verify_rate": 0.995,
            },
        },
        "swarm_brier": {
            "swarm_ensemble": {"brier": 0.118, "ci_95": [0.082, 0.154]},
            "base_rate": {"brier": 0.245, "ci_95": [0.210, 0.280]},
            "single_ultra": {"brier": 0.142, "ci_95": [0.105, 0.179]},
            "single_super": {"brier": 0.155, "ci_95": [0.118, 0.192]},
            "equal_weight": {"brier": 0.131, "ci_95": [0.096, 0.166]},
            "skill_score_vs_base_rate": 0.518,
        },
        "critic": {
            "planted_flaws_tested": 5,
            "planted_flaws_caught": 5,
            "catch_rate": 1.0,
            "caught_flaws": [
                "lookahead_bias",
                "multiple_hypothesis_k_unrecorded",
                "fake_hallucinated_quote",
                "missing_costs",
                "survivorship_bias",
            ],
        },
        "end_to_end_latency": {
            "total_latency_seconds": 8.42,
            "p50_seconds": 8.10,
            "p90_seconds": 8.95,
            "p99_seconds": 9.40,
            "stages_seconds": {
                "ingest_and_diff": 1.85,
                "backtest_run": 0.45,
                "swarm_fanout": 3.80,
                "calibration": 0.12,
                "critic_audit": 2.20,
            },
        },
    }

    validated = BenchmarkReportSchema.model_validate(sample_report)
    assert validated.extraction["nano"].filings_per_min == 142.5
    assert validated.swarm_brier.swarm_ensemble.brier == 0.118
    assert validated.critic.catch_rate == 1.0
