"""
Unit tests for Phase 2: Data layer (EDGAR fetcher, Item 1A parser, price loader).
Following AGENTS.md Rule 2: Tests written first.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import pytest

from tests.fixtures.sample_filings import SAMPLE_FILINGS
from data.sections import Item1AParser, FilingSection
from data.edgar import EDGARClient
from data.prices import PriceLoader


EASTERN = ZoneInfo("America/New_York")


def test_fixtures_3_tickers():
    """Verify fixtures cover at least 3 tickers."""
    tickers = {f["ticker"] for f in SAMPLE_FILINGS}
    assert len(tickers) >= 3, f"Expected at least 3 tickers, got {tickers}"
    assert "AAPL" in tickers
    assert "NVDA" in tickers
    assert "MSFT" in tickers


def test_section_parser_10_filings_gate():
    """
    Gate requirement: A section parser test on 10 filings with at least 9 parsed correctly.
    """
    parser = Item1AParser()
    parsed_count = 0
    total_filings = len(SAMPLE_FILINGS)
    assert total_filings == 10, f"Expected 10 sample filings, got {total_filings}"

    for filing in SAMPLE_FILINGS:
        section = parser.extract_item_1a(
            ticker=filing["ticker"],
            accession_number=filing["accession"],
            raw_text=filing["raw_text"],
            accepted_at_str=filing["accepted_at_str"],
        )
        if section is not None:
            # Check if expected risk factor content is present
            if filing["expected_risk_keyword"] in section.text:
                # Ensure section does not contain trailing item headers
                if "ITEM 1B" not in section.text.upper() and "ITEM 2" not in section.text.upper():
                    parsed_count += 1

    assert parsed_count >= 9, f"Gate failed: Only {parsed_count}/10 filings parsed correctly (required >= 9)"


def test_acceptance_timestamp_tz_aware_gate():
    """
    Gate requirement: Test that the acceptance timestamp is tz-aware.
    """
    parser = Item1AParser()
    sample = SAMPLE_FILINGS[0]
    section = parser.extract_item_1a(
        ticker=sample["ticker"],
        accession_number=sample["accession"],
        raw_text=sample["raw_text"],
        accepted_at_str=sample["accepted_at_str"],
    )
    assert section is not None
    assert section.accepted_at is not None
    assert section.accepted_at.tzinfo is not None, "Filing accepted_at must be timezone-aware!"

    # Verify timestamp parsing from SEC format "2023-11-03 18:04:45"
    expected_dt = datetime(2023, 11, 3, 18, 4, 45, tzinfo=EASTERN)
    assert section.accepted_at == expected_dt


def test_edgar_client_user_agent_and_rate_limiting():
    """Verify EDGAR client user agent compliance and rate limit enforcement."""
    client = EDGARClient(user_agent="ALBUS-Test/1.0 (researcher@example.com)")
    assert "researcher@example.com" in client.user_agent

    # Verify rate limit is configured to <= 10 req/s
    assert client.min_request_interval_seconds >= 0.10


def test_price_loader_next_open_execution_no_lookahead():
    """
    Verify trades execute at the next trading day's Open after SEC acceptance time.
    e.g. Filing accepted on Friday 2023-11-03 at 18:04 ET -> trade executes Monday 2023-11-06 Open.
    """
    # Sample synthetic daily bars
    dates = pd.date_range("2023-11-01", "2023-11-10", freq="B", tz=EASTERN)
    df = pd.DataFrame(
        {
            "Open": [170.0, 172.0, 175.0, 180.0, 182.0, 181.0, 183.0, 185.0],
            "High": [173.0, 176.0, 178.0, 183.0, 184.0, 183.0, 186.0, 187.0],
            "Low": [169.0, 171.0, 174.0, 179.0, 180.0, 179.0, 182.0, 184.0],
            "Close": [172.0, 175.0, 176.0, 182.0, 181.0, 182.0, 185.0, 186.0],
            "Adj Close": [172.0, 175.0, 176.0, 182.0, 181.0, 182.0, 185.0, 186.0],
            "Volume": [1000000] * 8,
        },
        index=dates,
    )

    loader = PriceLoader()
    loader.register_price_series("AAPL", df)

    # Acceptance time: Friday 2023-11-03 at 18:04:45 (after market close 16:00)
    acceptance_dt = datetime(2023, 11, 3, 18, 4, 45, tzinfo=EASTERN)

    # Must execute on Monday 2023-11-06 Open (180.0), NOT Friday Close (176.0)
    trade_time, exec_price = loader.get_execution_open("AAPL", acceptance_dt)

    assert trade_time.date() == datetime(2023, 11, 6).date()
    assert exec_price == 180.0


def test_price_loader_excess_returns():
    """Verify calculation of forward excess returns vs benchmark ETF."""
    dates = pd.date_range("2023-11-01", "2023-11-10", freq="B", tz=EASTERN)
    # Stock prices go from 100 to 110 (+10%)
    df_stock = pd.DataFrame(
        {"Open": [100.0, 102.0, 105.0, 108.0, 110.0, 110.0, 111.0, 112.0],
         "Adj Close": [100.0, 102.0, 105.0, 108.0, 110.0, 110.0, 111.0, 112.0]},
        index=dates,
    )
    # ETF benchmark prices go from 200 to 208 (+4%)
    df_etf = pd.DataFrame(
        {"Open": [200.0, 201.0, 203.0, 205.0, 208.0, 208.0, 209.0, 210.0],
         "Adj Close": [200.0, 201.0, 203.0, 205.0, 208.0, 208.0, 209.0, 210.0]},
        index=dates,
    )

    loader = PriceLoader()
    loader.register_price_series("NVDA", df_stock)
    loader.register_price_series("XLK", df_etf)

    start_dt = datetime(2023, 11, 1, 9, 30, tzinfo=EASTERN)
    # 4 trading days forward: from day 0 to day 4
    excess_ret = loader.calculate_excess_return(
        ticker="NVDA",
        benchmark_ticker="XLK",
        start_time=start_dt,
        trading_days=4,
    )
    # Stock return: (110 - 100)/100 = 0.10
    # ETF return: (208 - 200)/200 = 0.04
    # Excess: 0.10 - 0.04 = 0.06
    assert pytest.approx(excess_ret, abs=1e-4) == 0.06
