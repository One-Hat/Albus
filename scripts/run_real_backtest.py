"""
Real-Price Backtest Sanity Check (No LLM).
Evaluates a non-LLM signal (YoY change in Item 1A character length) using real SEC acceptance
timestamps and next-open price execution with transaction costs.
Outputs results/real_backtest_sanity.json and verifies no-lookahead execution.
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.backtest.engine import BacktestConfig, BacktestEngine, SignalRecord
from src.data.prices import PriceLoader


def run_sanity_backtest() -> Dict[str, Any]:
    validation_path = Path("results/real_data_validation.json")
    if not validation_path.exists():
        raise FileNotFoundError(f"Missing {validation_path}. Run validate_real_data.py first.")

    with open(validation_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    all_filings = val_data.get("filings", [])
    print(f"[INFO] Loaded {len(all_filings)} filings from {validation_path}")

    # Group filings by ticker and sort chronologically by accepted_at
    by_ticker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for filing in all_filings:
        by_ticker[filing["ticker"]].append(filing)

    price_loader = PriceLoader()

    # Pre-fetch prices for all tickers
    unique_tickers = sorted(by_ticker.keys())
    print(f"[INFO] Pre-fetching / verifying prices for {len(unique_tickers)} tickers...")
    for t in unique_tickers:
        try:
            df = price_loader.fetch_prices(t)
            print(f"  * {t}: {len(df)} daily price bars ({df.index[0].date()} to {df.index[-1].date()})")
        except Exception as e:
            print(f"  ! Warning: could not load prices for {t}: {e}")

    filings_used: List[Dict[str, Any]] = []
    filings_dropped: List[Dict[str, Any]] = []
    signals: List[SignalRecord] = []

    for ticker, filings in by_ticker.items():
        # Sort ascending by accepted_at
        filings.sort(key=lambda x: x["accepted_at"])

        for i, filing in enumerate(filings):
            acc_num = filing["accession_number"]
            accepted_str = filing["accepted_at"]
            curr_length = filing.get("section_length", 0)

            # Earliest filing has no baseline for YoY change
            if i == 0:
                filings_dropped.append({
                    "ticker": ticker,
                    "accession_number": acc_num,
                    "accepted_at": accepted_str,
                    "reason": "earliest_filing_no_prior_fiscal_year_baseline",
                })
                continue

            prev_filing = filings[i - 1]
            prev_length = prev_filing.get("section_length", 0)

            if prev_length <= 0 or curr_length <= 0:
                filings_dropped.append({
                    "ticker": ticker,
                    "accession_number": acc_num,
                    "accepted_at": accepted_str,
                    "reason": "invalid_or_zero_section_length",
                })
                continue

            # YoY percentage change in Item 1A character length
            yoy_change = (curr_length - prev_length) / float(prev_length)
            accepted_dt = datetime.fromisoformat(accepted_str)

            # Check if forward prices and execution open are available
            try:
                exec_time, exec_open = price_loader.get_execution_open(ticker, accepted_dt)
                fwd_ret = price_loader.calculate_forward_return(ticker, exec_time, trading_days=60)
            except Exception as exc:
                filings_dropped.append({
                    "ticker": ticker,
                    "accession_number": acc_num,
                    "accepted_at": accepted_str,
                    "reason": f"price_execution_unavailable: {exc}",
                })
                continue

            sig_record = SignalRecord(
                ticker=ticker,
                accepted_at=accepted_dt,
                signal=float(yoy_change),
            )
            signals.append(sig_record)
            filings_used.append({
                "ticker": ticker,
                "accession_number": acc_num,
                "accepted_at": accepted_str,
                "prev_length": prev_length,
                "curr_length": curr_length,
                "yoy_change": float(yoy_change),
                "execution_time": exec_time.isoformat(),
                "execution_open": exec_open,
                "forward_60d_return": float(fwd_ret),
            })

    print(f"\n[INFO] Signals constructed: {len(signals)} usable, {len(filings_dropped)} dropped")

    if not signals:
        raise RuntimeError("No usable signals could be constructed from real data.")

    # Run through backtest engine
    config = BacktestConfig(
        horizon_days=60,
        cost_bps=10.0,
        n_quantiles=10,
        periods_per_year=4.0,
        bootstrap_samples=1000,
        random_seed=42,
    )
    engine = BacktestEngine(price_loader=price_loader, config=config)
    result = engine.run(signals)

    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "yoy_item_1a_character_length_change",
        "description": "Non-LLM plumbing test: YoY change in Item 1A length executed next-open with 10bps costs",
        "config": {
            "horizon_days": config.horizon_days,
            "cost_bps": config.cost_bps,
            "n_quantiles": config.n_quantiles,
            "periods_per_year": config.periods_per_year,
            "bootstrap_samples": config.bootstrap_samples,
        },
        "metrics": {
            "sharpe": float(result.sharpe),
            "sharpe_ci_95": [float(result.sharpe_ci_95[0]), float(result.sharpe_ci_95[1])],
            "max_drawdown": float(result.max_drawdown),
            "turnover": float(result.turnover),
            "mean_period_return": float(result.mean_period_return),
            "total_return": float(result.total_return),
            "total_cost_drag": float(result.total_cost_drag),
            "sample_size": result.sample_size,
        },
        "counts": {
            "total_filings_evaluated": len(all_filings),
            "filings_used": len(filings_used),
            "filings_dropped": len(filings_dropped),
        },
        "filings_used": filings_used,
        "filings_dropped": filings_dropped,
    }

    out_path = Path("results/real_backtest_sanity.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Results written to {out_path.resolve()}")
    print("=" * 60)
    print("REAL PRICE BACKTEST SANITY SUMMARY")
    print("=" * 60)
    print(f"Strategy:            {output['strategy']}")
    print(f"Filings Evaluated:   {output['counts']['total_filings_evaluated']}")
    print(f"Filings Used:        {output['counts']['filings_used']}")
    print(f"Filings Dropped:     {output['counts']['filings_dropped']}")
    print(f"Sharpe Ratio:        {output['metrics']['sharpe']:.4f}")
    print(f"Sharpe 95% CI:       [{output['metrics']['sharpe_ci_95'][0]:.4f}, {output['metrics']['sharpe_ci_95'][1]:.4f}]")
    print(f"Max Drawdown:        {output['metrics']['max_drawdown']:.4f}")
    print(f"Turnover:            {output['metrics']['turnover']:.4f}")
    print(f"Total Return:        {output['metrics']['total_return']:.4f}")
    print(f"Total Cost Drag:     {output['metrics']['total_cost_drag']:.4f}")
    print("=" * 60)

    return output


if __name__ == "__main__":
    run_sanity_backtest()
