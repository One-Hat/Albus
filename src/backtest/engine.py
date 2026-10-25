"""
Walk-forward Decile Long/Short Backtest Engine (No LLM).
Enforces next-open trade execution after SEC acceptance timestamp, deducts transaction costs,
and computes Sharpe, max drawdown, turnover, and bootstrap confidence intervals.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, time as dtime
from typing import Dict, List, Literal, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo
import numpy as np
from pydantic import BaseModel, Field, field_validator

from backtest.metrics import (
    bootstrap_sharpe_ci,
    compute_max_drawdown,
    compute_sharpe,
    compute_turnover,
)
from data.prices import PriceLoader

EASTERN = ZoneInfo("America/New_York")
MARKET_OPEN_TIME = dtime(9, 30)


class LookaheadBiasError(ValueError):
    """Raised when a trade attempts to execute at or before the signal's acceptance timestamp."""


class SignalRecord(BaseModel):
    ticker: str
    accepted_at: datetime
    signal: float
    entry_time_override: Optional[datetime] = None

    @field_validator("accepted_at")
    @classmethod
    def validate_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("SignalRecord.accepted_at must be timezone-aware.")
        return v


class TradeExecution(BaseModel):
    ticker: str
    accepted_at: datetime
    entry_time: datetime
    entry_price: float
    side: Literal["long", "short"]
    weight: float
    gross_return: float
    net_return: float


class BacktestConfig(BaseModel):
    horizon_days: int = Field(default=60, ge=1)
    cost_bps: float = Field(default=10.0, ge=0.0)
    n_quantiles: int = Field(default=10, ge=2)
    periods_per_year: float = Field(default=4.0, gt=0.0)
    bootstrap_samples: int = Field(default=1000, ge=50)
    random_seed: int = 42


class BacktestResult(BaseModel):
    sharpe: float
    sharpe_ci_95: Tuple[float, float]
    max_drawdown: float
    turnover: float
    mean_period_return: float
    total_return: float
    total_cost_drag: float
    period_returns: List[float]
    sample_size: int
    costs_included: bool
    trades: List[TradeExecution]


class BacktestEngine:
    """
    Executes a decile (or quantile) long/short strategy on filing signals.
    Strictly enforces that every trade occurs at the NEXT market open after `accepted_at`.
    """

    def __init__(
        self,
        price_loader: PriceLoader,
        config: Optional[BacktestConfig] = None,
    ):
        self.price_loader = price_loader
        self.config = config or BacktestConfig()

    def _resolve_and_verify_entry(
        self,
        sig: SignalRecord,
    ) -> Tuple[datetime, float]:
        """
        Determine entry timestamp and open price, strictly rejecting any lookahead bias.
        """
        accepted_et = sig.accepted_at.astimezone(EASTERN)

        if sig.entry_time_override is not None:
            override_et = sig.entry_time_override.astimezone(EASTERN)
            if override_et <= accepted_et:
                raise LookaheadBiasError(
                    f"Lookahead detected for {sig.ticker}: entry_time {override_et} <= "
                    f"accepted_at {accepted_et}"
                )
            _, exec_price = self.price_loader.get_execution_open(sig.ticker, accepted_et)
            return override_et, exec_price

        exec_date_dt, exec_price = self.price_loader.get_execution_open(sig.ticker, accepted_et)
        # Attach exact 09:30 ET market open time to the trading day timestamp
        exec_open_dt = exec_date_dt.astimezone(EASTERN).replace(
            hour=MARKET_OPEN_TIME.hour,
            minute=MARKET_OPEN_TIME.minute,
            second=0,
            microsecond=0,
        )

        if exec_open_dt <= accepted_et:
            raise LookaheadBiasError(
                f"Lookahead detected for {sig.ticker}: computed open {exec_open_dt} <= "
                f"accepted_at {accepted_et}"
            )

        return exec_open_dt, exec_price

    def run(self, signals: Sequence[SignalRecord]) -> BacktestResult:
        if not signals:
            raise ValueError("Cannot run backtest on empty signals list.")

        # Group signals into cohorts by acceptance date
        cohorts: Dict[datetime, List[SignalRecord]] = defaultdict(list)
        for sig in signals:
            cohort_key = sig.accepted_at.astimezone(EASTERN)
            cohorts[cohort_key].append(sig)

        sorted_cohort_times = sorted(cohorts.keys())
        period_returns: List[float] = []
        weight_history: List[Dict[str, float]] = []
        all_trades: List[TradeExecution] = []
        total_cost_drag = 0.0

        cost_rate = self.config.cost_bps / 10_000.0
        prev_weights: Dict[str, float] = {}

        for cohort_dt in sorted_cohort_times:
            cohort_sigs = cohorts[cohort_dt]
            # Validate no-lookahead on every signal in the cohort before ranking
            resolved_entries: Dict[str, Tuple[datetime, float]] = {}
            for s in cohort_sigs:
                resolved_entries[s.ticker] = self._resolve_and_verify_entry(s)

            n = len(cohort_sigs)
            if n < 2:
                # Single-stock event: go long if signal >= 0 else short
                sig = cohort_sigs[0]
                entry_dt, entry_price = resolved_entries[sig.ticker]
                fwd_ret = self.price_loader.calculate_forward_return(
                    sig.ticker,
                    entry_dt,
                    trading_days=self.config.horizon_days,
                )
                side: Literal["long", "short"] = "long" if sig.signal >= 0 else "short"
                w = 1.0 if side == "long" else -1.0
                gross = w * fwd_ret
                net = gross - 2.0 * cost_rate
                total_cost_drag += 2.0 * cost_rate
                period_returns.append(net)
                weight_history.append({sig.ticker: w})
                all_trades.append(
                    TradeExecution(
                        ticker=sig.ticker,
                        accepted_at=sig.accepted_at,
                        entry_time=entry_dt,
                        entry_price=entry_price,
                        side=side,
                        weight=w,
                        gross_return=gross,
                        net_return=net,
                    )
                )
                continue

            # Sort cohort by signal ascending
            ordered = sorted(cohort_sigs, key=lambda s: s.signal)
            k = max(1, n // self.config.n_quantiles)
            short_bucket = ordered[:k]
            long_bucket = ordered[-k:]

            curr_weights: Dict[str, float] = {}
            for s in long_bucket:
                curr_weights[s.ticker] = 0.5 / len(long_bucket)
            for s in short_bucket:
                curr_weights[s.ticker] = -0.5 / len(short_bucket)

            # Compute one-way turnover for this rebalance
            all_tickers = set(prev_weights.keys()) | set(curr_weights.keys())
            step_turnover = 0.5 * sum(
                abs(curr_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_tickers
            )
            if not prev_weights:
                # Initial entry into positions incurs one-way cost on gross exposure
                step_turnover = sum(abs(w) for w in curr_weights.values())

            step_cost = step_turnover * 2.0 * cost_rate
            total_cost_drag += step_cost
            prev_weights = curr_weights
            weight_history.append(curr_weights)

            cohort_gross = 0.0
            selected = [(s, "long") for s in long_bucket] + [(s, "short") for s in short_bucket]
            per_trade_cost = step_cost / len(selected)

            for s, side_label in selected:
                entry_dt, entry_price = resolved_entries[s.ticker]
                fwd_ret = self.price_loader.calculate_forward_return(
                    s.ticker,
                    entry_dt,
                    trading_days=self.config.horizon_days,
                )
                w = curr_weights[s.ticker]
                trade_gross = w * fwd_ret * 2.0  # Normalized so +0.5 long and -0.5 short = 1x long-short spread
                trade_net = trade_gross - per_trade_cost
                cohort_gross += trade_gross

                all_trades.append(
                    TradeExecution(
                        ticker=s.ticker,
                        accepted_at=s.accepted_at,
                        entry_time=entry_dt,
                        entry_price=entry_price,
                        side=side_label,  # type: ignore[arg-type]
                        weight=w,
                        gross_return=trade_gross,
                        net_return=trade_net,
                    )
                )

            cohort_net = cohort_gross - step_cost
            period_returns.append(cohort_net)

        arr = np.asarray(period_returns, dtype=float)
        sharpe = compute_sharpe(arr, periods_per_year=self.config.periods_per_year)
        ci_95 = bootstrap_sharpe_ci(
            arr,
            periods_per_year=self.config.periods_per_year,
            n_boot=self.config.bootstrap_samples,
            seed=self.config.random_seed,
        )
        mdd = compute_max_drawdown(arr)
        turnover = compute_turnover(weight_history)
        mean_ret = float(np.mean(arr)) if arr.size > 0 else 0.0
        tot_ret = float(np.prod(1.0 + arr) - 1.0) if arr.size > 0 else 0.0

        return BacktestResult(
            sharpe=sharpe,
            sharpe_ci_95=ci_95,
            max_drawdown=mdd,
            turnover=turnover,
            mean_period_return=mean_ret,
            total_return=tot_ret,
            total_cost_drag=total_cost_drag,
            period_returns=period_returns,
            sample_size=len(all_trades),
            costs_included=self.config.cost_bps > 0.0,
            trades=all_trades,
        )
