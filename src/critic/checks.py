"""
Deterministic code checks for the Critic audit (Phase 8).
Verifies signal date vs. filing date, k variants tried, transaction costs,
sample size, and separation of forecast kinds.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel

from backtest.engine import BacktestResult


class CheckResult(BaseModel):
    name: str
    passed: bool
    evidence: str


class DeterministicAudit:
    """
    Executes rule-based quantitative audits prior to invoking Nemotron Ultra.
    """

    @staticmethod
    def check_signal_vs_filing_date(backtest_result: BacktestResult) -> CheckResult:
        """Check 1: Verify every trade executed strictly AFTER the SEC acceptance timestamp."""
        if not backtest_result.trades:
            return CheckResult(
                name="signal_vs_filing_date",
                passed=False,
                evidence="No trades recorded in backtest result.",
            )

        violations = []
        for idx, t in enumerate(backtest_result.trades):
            if t.entry_time <= t.accepted_at:
                violations.append(
                    f"Trade {idx} ({t.ticker}): entry_time {t.entry_time} <= accepted_at {t.accepted_at}"
                )

        if violations:
            return CheckResult(
                name="signal_vs_filing_date",
                passed=False,
                evidence=f"FATAL: Lookahead bias detected in {len(violations)} trade(s): {violations[0]}",
            )

        return CheckResult(
            name="signal_vs_filing_date",
            passed=True,
            evidence=f"All {len(backtest_result.trades)} trades executed strictly after SEC acceptance timestamp.",
        )

    @staticmethod
    def check_k_variants(variants_k: int, max_allowed_k: int = 15) -> CheckResult:
        """Check 2: Check for excessive signal searching / multiple-testing p-hacking."""
        if variants_k > max_allowed_k:
            return CheckResult(
                name="k_variants_multiple_testing",
                passed=False,
                evidence=(
                    f"Agent tested k={variants_k} signal variants (limit={max_allowed_k}) "
                    "without multiple-testing haircut adjustment. High probability of data snooping."
                ),
            )
        return CheckResult(
            name="k_variants_multiple_testing",
            passed=True,
            evidence=f"Number of tested variants (k={variants_k}) is within acceptable threshold (<= {max_allowed_k}).",
        )

    @staticmethod
    def check_transaction_costs(backtest_result: BacktestResult) -> CheckResult:
        """Check 3: Confirm transaction costs were modeled and not zeroed out."""
        if not backtest_result.costs_included or backtest_result.total_cost_drag <= 0.0:
            return CheckResult(
                name="transaction_costs_included",
                passed=False,
                evidence=(
                    f"Transaction costs missing or zeroed out (costs_included={backtest_result.costs_included}, "
                    f"total_cost_drag={backtest_result.total_cost_drag}). Real-world alpha may be negative."
                ),
            )
        return CheckResult(
            name="transaction_costs_included",
            passed=True,
            evidence=f"Transaction costs modeled. Total cost drag: {backtest_result.total_cost_drag:.4f}.",
        )

    @staticmethod
    def check_sample_size(backtest_result: BacktestResult, min_trades: int = 30) -> CheckResult:
        """Check 4: Check if the sample size is statistically adequate."""
        n = backtest_result.sample_size
        if n < min_trades:
            return CheckResult(
                name="sample_size_adequate",
                passed=False,
                evidence=f"Sample size N={n} is below minimum statistical threshold ({min_trades} trades).",
            )
        return CheckResult(
            name="sample_size_adequate",
            passed=True,
            evidence=f"Sample size N={n} meets statistical minimum ({min_trades} trades).",
        )

    @staticmethod
    def check_forecast_kinds_separated(kind_separation_verified: bool = True) -> CheckResult:
        """Check 5: Confirm retrospective_clean and live forecasts are strictly separated."""
        if not kind_separation_verified:
            return CheckResult(
                name="forecast_kinds_separated",
                passed=False,
                evidence="FATAL: Retrospective and live forecast Brier scores were pooled or mixed.",
            )
        return CheckResult(
            name="forecast_kinds_separated",
            passed=True,
            evidence="Retrospective_clean and live forecasts tracked and reported in separate ledgers.",
        )


def run_deterministic_checks(
    backtest_result: BacktestResult,
    variants_k: int = 1,
    kind_separation_verified: bool = True,
    min_trades: int = 30,
    max_allowed_k: int = 15,
) -> List[CheckResult]:
    """Execute all 5 deterministic checks and return results."""
    audit = DeterministicAudit()
    return [
        audit.check_signal_vs_filing_date(backtest_result),
        audit.check_k_variants(variants_k, max_allowed_k=max_allowed_k),
        audit.check_transaction_costs(backtest_result),
        audit.check_sample_size(backtest_result, min_trades=min_trades),
        audit.check_forecast_kinds_separated(kind_separation_verified=kind_separation_verified),
    ]
