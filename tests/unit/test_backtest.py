"""
Unit and Gate tests for Phase 4: Backtest engine (no LLM).
Following AGENTS.md Rule 2 (Write tests first) & Rule 9 (Keep test_no_lookahead green at all times).
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pytest

from data.prices import PriceLoader
from backtest.metrics import (
    bootstrap_sharpe_ci,
    compute_max_drawdown,
    compute_sharpe,
    compute_turnover,
)
from backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    LookaheadBiasError,
    SignalRecord,
)

EASTERN = ZoneInfo("America/New_York")


def _build_synthetic_universe(
    n_tickers: int = 20,
    n_periods: int = 24,
    horizon_days: int = 10,
    planted_alpha: float = 0.0,
    seed: int = 42,
):
    """
    Create a synthetic universe of stocks with periodic filing signals and daily Open/Close prices.
    If planted_alpha > 0, top signals earn +planted_alpha and bottom signals earn -planted_alpha.
    """
    rng = np.random.default_rng(seed)
    loader = PriceLoader()

    total_days = (n_periods + 2) * (horizon_days + 2)
    dates = pd.date_range("2021-01-04", periods=total_days, freq="B", tz=EASTERN)

    tickers = [f"SYN{i:02d}" for i in range(n_tickers)]
    signals_by_period = {}

    for p in range(n_periods):
        day_idx = p * horizon_days
        # Signal accepted at 16:30 ET (after market close) on day_idx
        filing_dt = dates[day_idx].to_pydatetime().replace(hour=16, minute=30, second=0)
        raw_scores = rng.standard_normal(n_tickers)
        signals_by_period[p] = (day_idx, filing_dt, raw_scores)

    all_signals = []
    for t_idx, ticker in enumerate(tickers):
        prices_open = [100.0]
        prices_close = [100.0]

        # Build step-by-step daily prices
        period_map = {
            signals_by_period[p][0] + 1: signals_by_period[p][2][t_idx]
            for p in range(n_periods)
        }
        active_signal = 0.0
        days_left = 0

        for d in range(1, total_days):
            if d in period_map:
                active_signal = period_map[d]
                days_left = horizon_days

            noise = rng.normal(0.0, 0.008)
            drift = (planted_alpha * np.sign(active_signal) / horizon_days) if days_left > 0 else 0.0
            days_left = max(0, days_left - 1)

            prev_close = prices_close[-1]
            # Overnight gap is small noise; intraday return carries the drift + noise
            open_p = prev_close * (1.0 + rng.normal(0.0, 0.001))
            close_p = open_p * (1.0 + drift + noise)
            prices_open.append(open_p)
            prices_close.append(close_p)

        df = pd.DataFrame(
            {
                "Open": prices_open,
                "High": np.maximum(prices_open, prices_close) * 1.005,
                "Low": np.minimum(prices_open, prices_close) * 0.995,
                "Close": prices_close,
                "Adj Close": prices_close,
                "Volume": [1_000_000] * total_days,
            },
            index=dates,
        )
        loader.register_price_series(ticker, df)

        for p in range(n_periods):
            day_idx, filing_dt, raw_scores = signals_by_period[p]
            all_signals.append(
                SignalRecord(
                    ticker=ticker,
                    accepted_at=filing_dt,
                    signal=float(raw_scores[t_idx]),
                )
            )

    return loader, all_signals


def test_metrics_unit():
    """Verify Sharpe, max drawdown, turnover, and bootstrap CI calculations."""
    returns = np.array([0.02, 0.03, 0.01, 0.025, 0.015, 0.02, 0.03, 0.02, 0.018, 0.022])
    sharpe = compute_sharpe(returns, periods_per_year=12)
    assert sharpe > 5.0

    # Equity curve with a 20% drawdown: 100 -> 120 -> 96 -> 130
    period_rets = np.array([0.20, -0.20, 0.354167])
    mdd = compute_max_drawdown(period_rets)
    assert pytest.approx(mdd, abs=1e-4) == 0.20

    # Turnover between two weight vectors
    w_prev = {"A": 0.5, "B": -0.5}
    w_curr = {"A": -0.5, "B": 0.5}
    turnover = compute_turnover([w_prev, w_curr])
    assert pytest.approx(turnover, abs=1e-4) == 1.0

    ci_low, ci_high = bootstrap_sharpe_ci(returns, periods_per_year=12, n_boot=500, seed=42)
    assert ci_low < sharpe < ci_high
    assert ci_low > 0.0


def test_planted_signal_recovered():
    """
    Gate requirement: On synthetic data a planted signal is recovered.
    """
    loader, signals = _build_synthetic_universe(
        n_tickers=20,
        n_periods=24,
        horizon_days=10,
        planted_alpha=0.05,  # 5% per 10-day period for top vs bottom
        seed=42,
    )
    config = BacktestConfig(
        horizon_days=10,
        cost_bps=10.0,
        n_quantiles=10,
        periods_per_year=25,
        bootstrap_samples=500,
        random_seed=42,
    )
    engine = BacktestEngine(price_loader=loader, config=config)
    result: BacktestResult = engine.run(signals)

    assert isinstance(result, BacktestResult)
    assert result.sharpe > 2.0, f"Expected strong positive Sharpe for planted signal, got {result.sharpe}"
    assert result.mean_period_return > 0.02
    assert result.sharpe_ci_95[0] > 0.5
    assert result.turnover >= 0.0
    assert result.total_cost_drag > 0.0


def test_pure_noise_gives_sharpe_near_zero():
    """
    Gate requirement: Pure noise gives a Sharpe near 0.
    """
    loader, signals = _build_synthetic_universe(
        n_tickers=20,
        n_periods=36,
        horizon_days=10,
        planted_alpha=0.0,  # Zero alpha (pure noise)
        seed=123,
    )
    config = BacktestConfig(
        horizon_days=10,
        cost_bps=5.0,
        n_quantiles=10,
        periods_per_year=25,
        bootstrap_samples=500,
        random_seed=123,
    )
    engine = BacktestEngine(price_loader=loader, config=config)
    result: BacktestResult = engine.run(signals)

    assert abs(result.sharpe) < 0.85, f"Expected Sharpe near 0 for pure noise, got {result.sharpe}"
    # 0 should fall within or near the 95% confidence interval
    assert result.sharpe_ci_95[0] <= 0.5 and result.sharpe_ci_95[1] >= -0.5


def test_no_lookahead():
    """
    Gate requirement & AGENTS.md Rule 9:
    A planted lookahead is caught by test_no_lookahead.
    """
    loader, signals = _build_synthetic_universe(
        n_tickers=20,
        n_periods=6,
        horizon_days=10,
        planted_alpha=0.04,
        seed=42,
    )
    engine = BacktestEngine(price_loader=loader, config=BacktestConfig(horizon_days=10))

    # 1. Verify normal execution has zero lookahead violations and every trade executes strictly after acceptance
    clean_result = engine.run(signals)
    assert len(clean_result.trades) > 0
    for trade in clean_result.trades:
        assert trade.entry_time > trade.accepted_at, (
            f"Lookahead violation: trade entry {trade.entry_time} <= acceptance {trade.accepted_at}"
        )

    # 2. Planted lookahead case A: explicit pre-acceptance entry_time override on a signal
    corrupted_signals = list(signals)
    first_sig = corrupted_signals[0]
    # Plant an entry_time_override that is BEFORE the SEC acceptance timestamp (same day 09:30 Open when filed at 16:30)
    planted_early_entry = first_sig.accepted_at.replace(hour=9, minute=30)
    corrupted_signals[0] = SignalRecord(
        ticker=first_sig.ticker,
        accepted_at=first_sig.accepted_at,
        signal=first_sig.signal,
        entry_time_override=planted_early_entry,
    )

    with pytest.raises(LookaheadBiasError):
        engine.run(corrupted_signals)

    # 3. Planted lookahead case B: naive (non-tz-aware) timestamp must be rejected
    with pytest.raises(Exception):
        SignalRecord(
            ticker="SYN00",
            accepted_at=datetime(2021, 1, 4, 16, 30),  # Naive datetime
            signal=1.0,
        )
