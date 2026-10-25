"""
Quantitative research tools for Nemotron Super:
list_filings, run_backtest, get_stats, and plot.
Maintains counter k of signal variants tested.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from backtest.engine import BacktestConfig, BacktestEngine, BacktestResult, SignalRecord
from data.sections import FilingSection
from extract.quote_check import FilingDiff


class ToolCallRequest(BaseModel):
    """Schema for LLM tool call or final synthesis decision."""
    action: Literal["call_tool", "final_answer"]
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[str] = None


class ResearchTools:
    """
    Stateful tool suite for the research agent.
    Tracks k (count of signal variants tested) and caches backtest results.
    """

    def __init__(
        self,
        filings: List[FilingSection],
        filing_diffs: List[FilingDiff],
        backtest_engine: BacktestEngine,
    ):
        self.filings = filings
        self.filing_diffs = filing_diffs
        self.backtest_engine = backtest_engine
        self.k: int = 0
        self.last_backtest_result: Optional[BacktestResult] = None
        self.backtest_history: List[Dict[str, Any]] = []

    def list_filings(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return metadata for all available filings in the research universe."""
        results = []
        for f in self.filings:
            if ticker and f.ticker.upper() != ticker.upper():
                continue
            results.append({
                "ticker": f.ticker,
                "accession_number": f.accession_number,
                "form_type": f.form_type,
                "accepted_at": f.accepted_at.isoformat(),
                "word_count": f.word_count,
            })
        return results

    def run_backtest(
        self,
        signal_definition: str,
        horizon_days: int = 60,
        cost_bps: float = 10.0,
        filter_keyword: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Formulate signals across filing diffs and execute a decile/quantile backtest.
        Increments variant counter k.
        """
        self.k += 1

        signals: List[SignalRecord] = []
        for diff in self.filing_diffs:
            score = 0.0
            if filter_keyword:
                kw = filter_keyword.lower()
                # If keyword matches any new risk claim, signal is -1.0 (bearish on added risk)
                has_kw = any(
                    kw in claim.text.lower() or kw in claim.quote.lower()
                    for claim in diff.new_risks
                )
                score = -1.0 if has_kw else 0.5
            else:
                # Default signal is the tone shift
                score = diff.tone_shift

            signals.append(
                SignalRecord(
                    ticker=diff.ticker,
                    accepted_at=diff.accepted_at,
                    signal=score,
                )
            )

        if not signals:
            return {"error": "No filing diffs available to generate signals."}

        # Override engine config with requested horizon and cost
        engine = BacktestEngine(
            price_loader=self.backtest_engine.price_loader,
            config=BacktestConfig(
                horizon_days=horizon_days,
                cost_bps=cost_bps,
                n_quantiles=self.backtest_engine.config.n_quantiles,
                periods_per_year=self.backtest_engine.config.periods_per_year,
                bootstrap_samples=self.backtest_engine.config.bootstrap_samples,
                random_seed=self.backtest_engine.config.random_seed,
            ),
        )

        result: BacktestResult = engine.run(signals)
        self.last_backtest_result = result

        summary = {
            "variant_k": self.k,
            "signal_definition": signal_definition,
            "horizon_days": horizon_days,
            "cost_bps": cost_bps,
            "sharpe": round(result.sharpe, 3),
            "sharpe_ci_95": [round(c, 3) for c in result.sharpe_ci_95],
            "max_drawdown": round(result.max_drawdown, 4),
            "turnover": round(result.turnover, 3),
            "total_return": round(result.total_return, 4),
            "mean_period_return": round(result.mean_period_return, 4),
            "sample_size": result.sample_size,
            "costs_included": result.costs_included,
        }
        self.backtest_history.append(summary)
        return summary

    def get_stats(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """Return in-depth distributional and execution metrics of the most recent backtest."""
        if self.last_backtest_result is None:
            return {"error": "No backtest has been executed yet. Call run_backtest first."}

        res = self.last_backtest_result
        data = {
            "variant_k": self.k,
            "sharpe": res.sharpe,
            "sharpe_ci_95": res.sharpe_ci_95,
            "max_drawdown": res.max_drawdown,
            "turnover": res.turnover,
            "mean_period_return": res.mean_period_return,
            "total_return": res.total_return,
            "total_cost_drag": res.total_cost_drag,
            "sample_size": res.sample_size,
            "number_of_periods": len(res.period_returns),
            "period_returns": [round(r, 4) for r in res.period_returns],
            "trade_count": len(res.trades),
        }
        if metric_name and metric_name in data:
            return {metric_name: data[metric_name]}
        return data

    def plot(self, chart_type: str = "cumulative_returns") -> Dict[str, Any]:
        """Generate structured chart coordinates for visualization in Streamlit / reports."""
        if self.last_backtest_result is None:
            return {"error": "No backtest has been executed yet. Call run_backtest first."}

        res = self.last_backtest_result
        if chart_type == "cumulative_returns":
            cum = 1.0
            points = [{"step": 0, "cumulative_return": 1.0}]
            for i, r in enumerate(res.period_returns, start=1):
                cum *= (1.0 + r)
                points.append({"step": i, "cumulative_return": round(cum, 4)})
            return {"chart_type": "cumulative_returns", "data": points}

        elif chart_type == "trades":
            trades_data = [
                {
                    "ticker": t.ticker,
                    "side": t.side,
                    "entry_time": t.entry_time.isoformat(),
                    "net_return": round(t.net_return, 4),
                }
                for t in res.trades
            ]
            return {"chart_type": "trades", "data": trades_data}

        return {"error": f"Unknown chart_type '{chart_type}'. Supported: 'cumulative_returns', 'trades'."}

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Route tool calls safely."""
        name = tool_name.strip().lower()
        if name == "list_filings":
            return self.list_filings(**arguments)
        elif name == "run_backtest":
            return self.run_backtest(**arguments)
        elif name == "get_stats":
            return self.get_stats(**arguments)
        elif name == "plot":
            return self.plot(**arguments)
        else:
            return {"error": f"Unknown tool '{tool_name}'. Allowed: list_filings, run_backtest, get_stats, plot."}

    def get_tool_descriptions(self) -> str:
        """Format tool specifications for LLM prompt context."""
        return (
            "Available tools:\n"
            "1. list_filings(ticker: Optional[str]) -> list of filing metadata\n"
            "2. run_backtest(signal_definition: str, horizon_days: int = 60, cost_bps: float = 10.0, filter_keyword: Optional[str]) -> backtest summary (increments variant counter k)\n"
            "3. get_stats(metric_name: Optional[str]) -> detailed stats on last backtest\n"
            "4. plot(chart_type: str = 'cumulative_returns') -> chart data series\n"
        )
