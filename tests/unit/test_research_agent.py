"""
Unit and Gate tests for Phase 5: Research agent (Super) and quantitative research tools.
Following AGENTS.md Rule 2: Tests written first.
"""
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import pytest

from llm.mock import MockLLM
from llm.client import LLMClient
from data.prices import PriceLoader
from data.sections import FilingSection
from extract.quote_check import Claim, FilingDiff
from backtest.engine import BacktestConfig, BacktestEngine
from research.tools import ResearchTools, ToolCallRequest
from research.agent import ResearchAgent, ResearchReport

EASTERN = ZoneInfo("America/New_York")


@pytest.fixture
def sample_research_environment():
    """Build a mini research environment with sample filings, diffs, and price series."""
    price_loader = PriceLoader()
    # Import synthetic price builder from backtest tests or build a quick 3-ticker series
    import pandas as pd
    dates = pd.date_range("2023-01-03", "2023-12-29", freq="B", tz=EASTERN)
    for ticker, base_p in [("AAPL", 150.0), ("NVDA", 200.0), ("MSFT", 250.0)]:
        n_days = len(dates)
        opens = [base_p + i * 0.2 for i in range(n_days)]
        closes = [base_p + i * 0.2 + 0.1 for i in range(n_days)]
        df = pd.DataFrame(
            {
                "Open": opens,
                "High": [p + 1.0 for p in opens],
                "Low": [p - 1.0 for p in opens],
                "Close": closes,
                "Adj Close": closes,
                "Volume": [500_000] * n_days,
            },
            index=dates,
        )
        price_loader.register_price_series(ticker, df)

    filings = [
        FilingSection(
            ticker="AAPL",
            accession_number="0000320193-23-000106",
            accepted_at=datetime(2023, 11, 3, 18, 4, tzinfo=EASTERN),
            text="Lawsuits and patent disputes have increased.",
            word_count=6,
        ),
        FilingSection(
            ticker="NVDA",
            accession_number="0001045810-23-000017",
            accepted_at=datetime(2023, 3, 3, 16, 30, tzinfo=EASTERN),
            text="Export controls on AI chips have expanded.",
            word_count=7,
        ),
        FilingSection(
            ticker="MSFT",
            accession_number="0000950170-23-035122",
            accepted_at=datetime(2023, 7, 27, 16, 10, tzinfo=EASTERN),
            text="Cloud infrastructure cyberattacks pose risks.",
            word_count=5,
        ),
    ]

    diffs = [
        FilingDiff(
            ticker="AAPL",
            accepted_at=datetime(2023, 11, 3, 18, 4, tzinfo=EASTERN),
            new_risks=[
                Claim(
                    text="New lawsuit disclosures",
                    quote="Lawsuits and patent disputes have increased.",
                    source_id="AAPL_2023",
                )
            ],
            removed_risks=[],
            tone_shift=-0.4,
        ),
        FilingDiff(
            ticker="NVDA",
            accepted_at=datetime(2023, 3, 3, 16, 30, tzinfo=EASTERN),
            new_risks=[
                Claim(
                    text="Export control restrictions",
                    quote="Export controls on AI chips have expanded.",
                    source_id="NVDA_2023",
                )
            ],
            removed_risks=[],
            tone_shift=-0.2,
        ),
        FilingDiff(
            ticker="MSFT",
            accepted_at=datetime(2023, 7, 27, 16, 10, tzinfo=EASTERN),
            new_risks=[
                Claim(
                    text="Cloud cyber risks",
                    quote="Cloud infrastructure cyberattacks pose risks.",
                    source_id="MSFT_2023",
                )
            ],
            removed_risks=[],
            tone_shift=-0.1,
        ),
    ]

    backtest_engine = BacktestEngine(
        price_loader=price_loader,
        config=BacktestConfig(horizon_days=5, cost_bps=5.0),
    )
    tools = ResearchTools(
        filings=filings,
        filing_diffs=diffs,
        backtest_engine=backtest_engine,
    )
    return tools


def test_research_tools_execution(sample_research_environment):
    """Verify individual tool executions: list_filings, run_backtest, get_stats, plot."""
    tools: ResearchTools = sample_research_environment

    # 1. list_filings
    filings_res = tools.list_filings()
    assert len(filings_res) == 3
    assert any(f["ticker"] == "AAPL" for f in filings_res)

    # 2. run_backtest
    bt_res = tools.run_backtest(
        signal_definition="lawsuit",
        filter_keyword="lawsuit",
        horizon_days=5,
    )
    assert tools.k == 1
    assert "sharpe" in bt_res
    assert "sample_size" in bt_res
    assert bt_res["sample_size"] == 3

    # 3. get_stats
    stats_res = tools.get_stats()
    assert "sharpe" in stats_res
    assert "sharpe_ci_95" in stats_res
    assert "max_drawdown" in stats_res

    # 4. plot
    plot_res = tools.plot(chart_type="cumulative_returns")
    assert "chart_type" in plot_res
    assert "data" in plot_res
    assert len(plot_res["data"]) > 0


@pytest.mark.asyncio
async def test_gate_scripted_hypothesis_runs_end_to_end_and_k_recorded(sample_research_environment):
    """
    Gate requirement: A scripted hypothesis runs end to end, k is recorded.
    """
    tools = sample_research_environment
    mock = MockLLM()

    # Sequence of Super tool calls:
    # 1. Calls list_filings
    # 2. Calls run_backtest with variant 1 (lawsuits)
    # 3. Calls run_backtest with variant 2 (tone shift)
    # 4. Calls get_stats
    # 5. Calls plot
    # 6. Returns final research synthesis
    responses = [
        {
            "content": json.dumps({
                "action": "call_tool",
                "tool_name": "list_filings",
                "arguments": {},
            }),
            "reasoning": "First, let's explore available filings to see what companies and dates we have.",
        },
        {
            "content": json.dumps({
                "action": "call_tool",
                "tool_name": "run_backtest",
                "arguments": {"signal_definition": "lawsuit_disclosures", "filter_keyword": "lawsuit"},
            }),
            "reasoning": "Now testing variant 1: companies with sudden new lawsuit disclosures.",
        },
        {
            "content": json.dumps({
                "action": "call_tool",
                "tool_name": "run_backtest",
                "arguments": {"signal_definition": "negative_tone_shift", "filter_keyword": None},
            }),
            "reasoning": "Now testing variant 2: negative overall tone shift.",
        },
        {
            "content": json.dumps({
                "action": "call_tool",
                "tool_name": "get_stats",
                "arguments": {},
            }),
            "reasoning": "Inspecting distributional metrics for the best variant.",
        },
        {
            "content": json.dumps({
                "action": "call_tool",
                "tool_name": "plot",
                "arguments": {"chart_type": "cumulative_returns"},
            }),
            "reasoning": "Generating cumulative returns plot.",
        },
        {
            "content": json.dumps({
                "action": "final_answer",
                "summary": (
                    "Tested 2 signal variants. Sudden lawsuit additions exhibited a negative forward "
                    "excess return, consistent with the initial hypothesis. Transaction costs were modeled."
                ),
            }),
            "reasoning": "Synthesizing research conclusion with citations and stats.",
        },
    ]
    mock.register_sequence("super", responses)

    llm_client = LLMClient(mock_provider=mock)
    agent = ResearchAgent(llm_client=llm_client, tools=tools, max_iterations=10)

    report: ResearchReport = await agent.run_hypothesis(
        "Do companies that suddenly mention new lawsuit worries underperform the market?"
    )

    assert isinstance(report, ResearchReport)
    assert report.variants_tested_k == 2, f"Expected k=2, got {report.variants_tested_k}"
    assert "lawsuit" in report.summary_narrative.lower()
    assert report.final_backtest is not None
    assert len(report.trace) >= 6


@pytest.mark.asyncio
async def test_gate_loop_stops_cleanly_when_cap_is_hit(sample_research_environment):
    """
    Gate requirement: The loop stops cleanly when the max-iterations cap is hit.
    """
    tools = sample_research_environment
    mock = MockLLM()

    # Super attempts to run backtests forever
    endless_responses = [
        {
            "content": json.dumps({
                "action": "call_tool",
                "tool_name": "run_backtest",
                "arguments": {"signal_definition": f"variant_{i}"},
            }),
            "reasoning": f"Endless variant exploration {i}",
        }
        for i in range(20)
    ]
    mock.register_sequence("super", endless_responses)

    llm_client = LLMClient(mock_provider=mock)
    max_cap = 3
    agent = ResearchAgent(llm_client=llm_client, tools=tools, max_iterations=max_cap)

    report: ResearchReport = await agent.run_hypothesis("Testing iteration cap behavior")

    assert report.iterations == max_cap, f"Expected loop to stop at cap {max_cap}, got {report.iterations}"
    assert report.variants_tested_k == max_cap
    assert "hit maximum iteration cap" in report.summary_narrative.lower()
