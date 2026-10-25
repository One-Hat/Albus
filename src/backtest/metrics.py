"""
Quantitative performance metrics for the backtest engine:
Sharpe ratio, maximum drawdown, portfolio turnover, and bootstrap confidence intervals.
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np


def compute_sharpe(
    returns: Sequence[float] | np.ndarray,
    periods_per_year: float = 252.0,
) -> float:
    """Compute annualized Sharpe ratio (assuming zero risk-free rate on excess/long-short returns)."""
    arr = np.asarray(returns, dtype=float)
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std <= 1e-12:
        return 0.0
    mean = float(np.mean(arr))
    return float((mean / std) * np.sqrt(periods_per_year))


def compute_max_drawdown(returns: Sequence[float] | np.ndarray) -> float:
    """
    Compute maximum drawdown from a sequence of periodic returns.
    Returns a positive fraction (e.g., 0.20 for a 20% max drawdown).
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    equity = np.cumprod(np.concatenate([[1.0], 1.0 + arr]))
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / np.maximum(running_max, 1e-12)
    return float(np.max(drawdowns))


def compute_turnover(weight_history: List[Dict[str, float]]) -> float:
    """
    Compute average one-way turnover per rebalance across a sequence of portfolio weight dicts.
    One-way turnover at step t = 0.5 * sum_i |w_{i,t} - w_{i,t-1}|.
    """
    if len(weight_history) < 2:
        return 0.0

    turnovers: List[float] = []
    for idx in range(1, len(weight_history)):
        w_prev = weight_history[idx - 1]
        w_curr = weight_history[idx]
        all_tickers = set(w_prev.keys()) | set(w_curr.keys())
        l1_diff = sum(abs(w_curr.get(t, 0.0) - w_prev.get(t, 0.0)) for t in all_tickers)
        turnovers.append(0.5 * l1_diff)

    return float(np.mean(turnovers))


def bootstrap_sharpe_ci(
    returns: Sequence[float] | np.ndarray,
    periods_per_year: float = 252.0,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the annualized Sharpe ratio.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size < 2:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    n = arr.size
    boot_sharpes = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        boot_sharpes[b] = compute_sharpe(sample, periods_per_year=periods_per_year)

    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(boot_sharpes, alpha))
    high = float(np.quantile(boot_sharpes, 1.0 - alpha))
    return (low, high)
