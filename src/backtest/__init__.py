"""
Backtest engine and quantitative performance metrics module.
"""
from backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    LookaheadBiasError,
    SignalRecord,
    TradeExecution,
)
from backtest.metrics import (
    bootstrap_sharpe_ci,
    compute_max_drawdown,
    compute_sharpe,
    compute_turnover,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "LookaheadBiasError",
    "SignalRecord",
    "TradeExecution",
    "bootstrap_sharpe_ci",
    "compute_max_drawdown",
    "compute_sharpe",
    "compute_turnover",
]
