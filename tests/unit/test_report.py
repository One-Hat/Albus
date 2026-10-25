"""
Unit and Gate tests for Phase 9: Citation integrity and report builder.
Following AGENTS.md Rule 2 (Tests written first) & Rule 6 (Verbatim quote verification).
"""
import pytest
from extract.quote_check import Claim
from report.builder import CitationIntegrityError, ReportBuilder, ResearchReport


@pytest.fixture
def sample_filing_fixtures():
    return {
        "AAPL_2023": (
            "The Company's business, reputation, results of operations, financial condition and "
            "stock price can be affected by a number of factors, including global economic conditions, "
            "supply chain disruptions, and geopolitical tensions in international trade."
        ),
        "NVDA_2024": (
            "We face intense competition in AI computing platforms and graphic processors. Failure to "
            "develop and commercialize new architectures on schedule could harm our competitive position."
        ),
    }


def test_report_builder_clean_case(sample_filing_fixtures):
    """
    Gate test: Build report from fixtures and assert every claim links to a quote
    that is a verbatim substring of its source filing.
    """
    builder = ReportBuilder(source_filings=sample_filing_fixtures)

    claims = [
        Claim(
            text="Geopolitical risks impacting global supply chains",
            quote="supply chain disruptions, and geopolitical tensions in international trade.",
            source_id="AAPL_2023",
        ),
        Claim(
            text="AI platform architecture competition",
            quote="We face intense competition in AI computing platforms and graphic processors.",
            source_id="NVDA_2024",
        ),
    ]

    report: ResearchReport = builder.build_report(
        ticker="AAPL",
        title="Fiscal Year Risk Factor Assessment",
        claims=claims,
        strict=True,
    )

    assert report.ticker == "AAPL"
    assert report.dropped_claims_count == 0
    assert len(report.verified_claims) == 2

    # Assert every claim links to a quote that is a verbatim substring of its source filing
    for c in report.verified_claims:
        source_text = sample_filing_fixtures[c.source_id]
        assert c.quote in source_text, f"Claim quote '{c.quote}' was not a verbatim substring of {c.source_id}!"

    assert "[VERIFIED CITATION]" in report.markdown_content
    assert "Geopolitical risks" in report.markdown_content
    assert "AI platform architecture" in report.markdown_content


def test_planted_fake_quote_fails_build(sample_filing_fixtures):
    """
    Gate requirement: A planted fake quote must fail the build when strict=True.
    """
    builder = ReportBuilder(source_filings=sample_filing_fixtures)

    valid_claim = Claim(
        text="Supply chain disruption",
        quote="supply chain disruptions, and geopolitical tensions",
        source_id="AAPL_2023",
    )
    # Planted fake quote: completely fabricated, does not exist in AAPL_2023
    planted_fake_claim = Claim(
        text="Fabricated dividend announcement",
        quote="The Board of Directors has guaranteed an extraordinary cash distribution of $100 per share.",
        source_id="AAPL_2023",
    )

    with pytest.raises(CitationIntegrityError) as exc_info:
        builder.build_report(
            ticker="AAPL",
            title="Rigged Report",
            claims=[valid_claim, planted_fake_claim],
            strict=True,
        )

    assert "Citation integrity failure" in str(exc_info.value)
    assert "guaranteed an extraordinary cash distribution" in str(exc_info.value)


def test_runtime_drops_and_counts_unverifiable_claims(sample_filing_fixtures):
    """
    Gate requirement: At runtime (strict=False), unverifiable claims are dropped and counted.
    """
    builder = ReportBuilder(source_filings=sample_filing_fixtures)

    claims = [
        Claim(
            text="Legitimate AI competition risk",
            quote="We face intense competition in AI computing platforms and graphic processors.",
            source_id="NVDA_2024",
        ),
        Claim(
            text="Unverifiable claim 1 (hallucinated quote)",
            quote="Management expects revenue to double every six months indefinitely.",
            source_id="NVDA_2024",
        ),
        Claim(
            text="Unverifiable claim 2 (unregistered source)",
            quote="Some quote from an unknown document",
            source_id="UNKNOWN_DOC_999",
        ),
    ]

    report: ResearchReport = builder.build_report(
        ticker="NVDA",
        title="Runtime Filtered Report",
        claims=claims,
        strict=False,
    )

    # Asserts that unverifiable claims are dropped and counted
    assert report.dropped_claims_count == 2
    assert len(report.verified_claims) == 1
    assert report.verified_claims[0].text == "Legitimate AI competition risk"

    # Verify dropped claims metadata
    dropped_quotes = [d["quote"] for d in report.dropped_claims]
    assert "Management expects revenue to double" in dropped_quotes[0]
    assert "Some quote from an unknown document" in dropped_quotes[1]

    # Verify markdown report does not present the fake claims as evidence
    assert "Legitimate AI competition risk" in report.markdown_content
    assert "Management expects revenue to double" not in report.markdown_content.split("## 1. Verified Key Risk Disclosures")[1].split("## 4. Citation Integrity Audit")[0]
    assert "## 4. Citation Integrity Audit (Dropped Unverified Claims)" in report.markdown_content
