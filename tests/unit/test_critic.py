"""
Unit and Gate tests for Phase 8: Critic (Ultra) and deterministic checks.
Following AGENTS.md Rule 2: Tests written first.
"""
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import pytest

from llm.mock import MockLLM
from llm.client import LLMClient
from backtest.engine import BacktestConfig, BacktestResult, TradeExecution
from critic.checks import CheckResult, DeterministicAudit, run_deterministic_checks
from critic.critic import CriticVerdict, UltraCritic

EASTERN = ZoneInfo("America/New_York")


def _make_clean_backtest_result(sample_size: int = 40, cost_bps: float = 10.0) -> BacktestResult:
    """Construct a clean, valid BacktestResult with no lookahead, sufficient sample, and costs."""
    accepted = datetime(2023, 11, 3, 18, 4, tzinfo=EASTERN)
    entry = datetime(2023, 11, 6, 9, 30, tzinfo=EASTERN)  # Next Monday open

    trades = [
        TradeExecution(
            ticker=f"TICK{i:02d}",
            accepted_at=accepted,
            entry_time=entry,
            entry_price=100.0,
            side="long" if i % 2 == 0 else "short",
            weight=0.05,
            gross_return=0.03,
            net_return=0.03 - (cost_bps / 10000.0),
        )
        for i in range(sample_size)
    ]
    return BacktestResult(
        sharpe=1.85,
        sharpe_ci_95=(1.10, 2.60),
        max_drawdown=0.08,
        turnover=0.45,
        mean_period_return=0.025,
        total_return=0.22,
        total_cost_drag=0.015,
        period_returns=[0.02, 0.03, -0.01, 0.04, 0.02, 0.01, 0.03, 0.02],
        sample_size=sample_size,
        costs_included=cost_bps > 0.0,
        trades=trades,
    )


def test_deterministic_checks_clean_case():
    """Verify clean case passes all 5 deterministic checks."""
    clean_bt = _make_clean_backtest_result()
    results = run_deterministic_checks(
        backtest_result=clean_bt,
        variants_k=2,
        kind_separation_verified=True,
    )
    assert len(results) == 5
    assert all(c.passed for c in results), f"Expected all checks to pass, got: {results}"


@pytest.mark.asyncio
async def test_gate_planted_flaws_caught_and_clean_case_passes():
    """
    Gate requirement:
    On 5 planted-flaw cases the critic catches at least 4, and it does not reject a clean case.
    """
    mock = MockLLM()

    # Register ultra responses for:
    # 1. Clean case -> pass
    # 2. Planted Flaw 1 (Lookahead bias) -> reject
    # 3. Planted Flaw 2 (Zero costs) -> reject
    # 4. Planted Flaw 3 (Sample size too small) -> reject
    # 5. Planted Flaw 4 (Excessive k variants / p-hacking) -> weak or reject
    # 6. Planted Flaw 5 (Mixed forecast kinds) -> reject
    responses = [
        # Clean
        {
            "content": json.dumps({
                "verdict": "pass",
                "narrative": "Rigorous backtest. No lookahead detected, transaction costs modeled, adequate sample.",
            }),
            "reasoning": "Scrutinized timestamps, costs, sample size, and k=2 variants.",
        },
        # Flaw 1: Lookahead
        {
            "content": json.dumps({
                "verdict": "reject",
                "narrative": "FATAL: Lookahead bias detected. Trades entered before filing acceptance.",
            }),
            "reasoning": "Detected entry_time <= accepted_at timestamp violation.",
        },
        # Flaw 2: Zero costs
        {
            "content": json.dumps({
                "verdict": "reject",
                "narrative": "FATAL: Backtest omitted transaction costs. Alpha is likely wiped out by slippage.",
            }),
            "reasoning": "Detected costs_included=False / zero cost drag.",
        },
        # Flaw 3: Small sample
        {
            "content": json.dumps({
                "verdict": "reject",
                "narrative": "REJECT: Sample size N=6 is statistically insufficient to distinguish skill from luck.",
            }),
            "reasoning": "Sample size below minimum threshold.",
        },
        # Flaw 4: k variants too high
        {
            "content": json.dumps({
                "verdict": "reject",
                "narrative": "REJECT: Data snooping. Agent tested k=28 signal variants without haircut adjustment.",
            }),
            "reasoning": "Excessive multiple testing detected.",
        },
        # Flaw 5: Mixed kinds
        {
            "content": json.dumps({
                "verdict": "reject",
                "narrative": "FATAL: Retrospective and live forecast Brier scores were pooled together.",
            }),
            "reasoning": "Violation of forecast kind separation rule.",
        },
    ]
    mock.register_sequence("ultra", responses)

    llm_client = LLMClient(mock_provider=mock)
    critic = UltraCritic(llm_client=llm_client)

    # 1. Clean Case
    clean_bt = _make_clean_backtest_result(sample_size=50, cost_bps=10.0)
    clean_verdict: CriticVerdict = await critic.audit(
        hypothesis="Do new lawsuit risk factor disclosures cause stock underperformance?",
        backtest_result=clean_bt,
        variants_k=2,
        kind_separation_verified=True,
    )
    assert clean_verdict.verdict == "pass", f"Clean case must pass, got {clean_verdict.verdict}"
    assert all(c.passed for c in clean_verdict.checks)

    caught_flaws = 0

    # Planted Flaw 1: Lookahead bias (entry_time before accepted_at)
    flaw1_bt = _make_clean_backtest_result()
    flaw1_bt.trades[0].entry_time = flaw1_bt.trades[0].accepted_at.replace(hour=9, minute=30)  # same day 09:30 vs 18:04
    v1 = await critic.audit(
        hypothesis="Hypothesis 1",
        backtest_result=flaw1_bt,
        variants_k=2,
        kind_separation_verified=True,
    )
    if v1.verdict == "reject" and any(c.name == "signal_vs_filing_date" and not c.passed for c in v1.checks):
        caught_flaws += 1

    # Planted Flaw 2: Zero transaction costs
    flaw2_bt = _make_clean_backtest_result(cost_bps=0.0)
    flaw2_bt.costs_included = False
    v2 = await critic.audit(
        hypothesis="Hypothesis 2",
        backtest_result=flaw2_bt,
        variants_k=2,
        kind_separation_verified=True,
    )
    if v2.verdict == "reject" and any(c.name == "transaction_costs_included" and not c.passed for c in v2.checks):
        caught_flaws += 1

    # Planted Flaw 3: Sample size too small (N = 6 < 30)
    flaw3_bt = _make_clean_backtest_result(sample_size=6)
    v3 = await critic.audit(
        hypothesis="Hypothesis 3",
        backtest_result=flaw3_bt,
        variants_k=2,
        kind_separation_verified=True,
    )
    if v3.verdict in ("reject", "weak") and any(c.name == "sample_size_adequate" and not c.passed for c in v3.checks):
        caught_flaws += 1

    # Planted Flaw 4: k variants tried is excessive (k = 28 > 15)
    flaw4_bt = _make_clean_backtest_result()
    v4 = await critic.audit(
        hypothesis="Hypothesis 4",
        backtest_result=flaw4_bt,
        variants_k=28,
        kind_separation_verified=True,
    )
    if v4.verdict in ("reject", "weak") and any(c.name == "k_variants_multiple_testing" and not c.passed for c in v4.checks):
        caught_flaws += 1

    # Planted Flaw 5: Mixed forecast kinds (kind_separation_verified = False)
    flaw5_bt = _make_clean_backtest_result()
    v5 = await critic.audit(
        hypothesis="Hypothesis 5",
        backtest_result=flaw5_bt,
        variants_k=2,
        kind_separation_verified=False,
    )
    if v5.verdict == "reject" and any(c.name == "forecast_kinds_separated" and not c.passed for c in v5.checks):
        caught_flaws += 1

    assert caught_flaws >= 4, f"Gate failed: Critic caught {caught_flaws}/5 planted flaws (required >= 4)"
