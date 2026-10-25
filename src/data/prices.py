"""
Price Data Loader with Split Adjustments and No-Lookahead Trade Execution.
Trades execute strictly at the NEXT trading day's OPEN following the SEC acceptance timestamp.
"""
from __future__ import annotations
import os
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf

EASTERN = ZoneInfo("America/New_York")
MARKET_OPEN_TIME = dtime(9, 30)
MARKET_CLOSE_TIME = dtime(16, 0)


class PriceLoader:
    """
    Loads daily historical prices, handles stock split adjustments,
    and calculates excess returns strictly preventing lookahead bias.
    """

    def __init__(self, cache_dir: str | Path = "data/cache/prices"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, pd.DataFrame] = {}

    def register_price_series(self, ticker: str, df: pd.DataFrame) -> None:
        """Register an in-memory price DataFrame for offline testing or caching."""
        clean_df = df.copy()
        # Ensure index is timezone-aware in Eastern Time
        if clean_df.index.tz is None:
            clean_df.index = clean_df.index.tz_localize(EASTERN)
        else:
            clean_df.index = clean_df.index.tz_convert(EASTERN)
        self._registry[ticker.upper()] = clean_df

    def fetch_prices(
        self,
        ticker: str,
        start_date: str = "2018-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch daily price data using yfinance (auto_adjust=True for splits & dividends).
        Falls back to local cache/registry if network is unavailable.
        """
        symbol = ticker.upper()
        if symbol in self._registry:
            return self._registry[symbol]

        cache_path = self.cache_dir / f"{symbol}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            self._registry[symbol] = df
            return df

        try:
            # yfinance auto_adjust=True correctly accounts for splits and dividends
            ticker_obj = yf.Ticker(symbol)
            df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
            if df.empty:
                raise ValueError(f"No price data found for ticker: {symbol}")

            if df.index.tz is None:
                df.index = df.index.tz_localize(EASTERN)
            else:
                df.index = df.index.tz_convert(EASTERN)

            # Cache to disk
            df.to_parquet(cache_path)
            self._registry[symbol] = df
            return df
        except Exception as e:
            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                self._registry[symbol] = df
                return df
            raise RuntimeError(f"Failed to fetch prices for {symbol}: {e}") from e

    def get_execution_open(
        self,
        ticker: str,
        filing_accepted_at: datetime,
    ) -> Tuple[datetime, float]:
        """
        Determine the exact trade execution time and Open price.
        AGENTS.md Rule 9 & Spec Decision 3:
        Signal date is SEC acceptance timestamp. Trades occur at the NEXT MARKET OPEN.
        - If filed after 09:30 on day T, trade executes at day T+1 Open.
        - If filed before 09:30 on day T, trade executes at day T Open.
        """
        prices = self.fetch_prices(ticker)
        filing_dt = filing_accepted_at.astimezone(EASTERN)
        filing_date = filing_dt.date()
        filing_time = filing_dt.time()

        # If filed before market open (09:30 ET), day T Open is available
        if filing_time < MARKET_OPEN_TIME:
            eligible_rows = prices[prices.index.date >= filing_date]
        else:
            # Filed during or after market hours: trade happens at next day's open
            eligible_rows = prices[prices.index.date > filing_date]

        if eligible_rows.empty:
            raise ValueError(
                f"No future price data available for {ticker} after filing timestamp {filing_accepted_at}"
            )

        exec_row = eligible_rows.iloc[0]
        exec_time = eligible_rows.index[0]
        exec_price = float(exec_row["Open"])
        return exec_time, exec_price

    def calculate_forward_return(
        self,
        ticker: str,
        start_time: datetime,
        trading_days: int = 60,
    ) -> float:
        """
        Calculate forward return from the execution Open over N trading days.
        """
        prices = self.fetch_prices(ticker)
        start_dt = start_time.astimezone(EASTERN)
        start_date = start_dt.date()

        eligible = prices[prices.index.date >= start_date]
        if len(eligible) <= trading_days:
            # Use max available horizon if period is short
            end_row = eligible.iloc[-1]
        else:
            end_row = eligible.iloc[trading_days]

        start_row = eligible.iloc[0]
        start_price = float(start_row["Open"])
        end_price = float(end_row["Close"] if "Close" in end_row else end_row["Adj Close"])

        return (end_price - start_price) / start_price

    def calculate_excess_return(
        self,
        ticker: str,
        benchmark_ticker: str,
        start_time: datetime,
        trading_days: int = 60,
    ) -> float:
        """
        Calculate forward excess return: stock_return - benchmark_return.
        e.g. 60-trading-day excess return vs sector ETF.
        """
        stock_ret = self.calculate_forward_return(ticker, start_time, trading_days)
        bench_ret = self.calculate_forward_return(benchmark_ticker, start_time, trading_days)
        return stock_ret - bench_ret
