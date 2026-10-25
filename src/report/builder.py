"""
Quantitative Research Report Builder with Strict & Runtime Citation Verification (Phase 9 Gate).
AGENTS.md Rule 6: Every LLM output is validated by pydantic. Quotes must be verbatim substrings.
Enforces that every claim in the generated research report maps to a verifiable verbatim quote.
In strict mode, an unverified quote fails the build immediately.
At runtime, unverifiable claims are dropped and counted in audit metadata.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from extract.quote_check import Claim, is_verbatim_quote


class CitationIntegrityError(ValueError):
    """Raised when strict citation verification fails on an unverified quote."""


class ResearchReport(BaseModel):
    ticker: str
    title: str
    generated_at: datetime
    verified_claims: List[Claim]
    dropped_claims_count: int = Field(ge=0)
    dropped_claims: List[Dict[str, str]]
    markdown_content: str
    swarm_forecast: Optional[Dict[str, Any]] = None
    backtest_metrics: Optional[Dict[str, Any]] = None


class ReportBuilder:
    """
    Builds structured markdown research reports with verifiable citations.
    Integrates verbatim text checks against source 10-K sections.
    """

    def __init__(self, source_filings: Optional[Dict[str, str]] = None):
        """
        source_filings: Mapping from source_id (e.g. 'AAPL_2023_10K') to source filing text.
        """
        self.source_filings: Dict[str, str] = source_filings or {}

    def register_source_filing(self, source_id: str, text: str) -> None:
        """Register or update source filing text for citation verification."""
        self.source_filings[source_id] = text

    def verify_claim(self, claim: Claim) -> bool:
        """Check if claim.quote is a verbatim substring of registered source text."""
        source_text = self.source_filings.get(claim.source_id)
        if not source_text:
            return False
        return is_verbatim_quote(claim.quote, source_text)

    def build_report(
        self,
        ticker: str,
        title: str,
        claims: List[Claim],
        strict: bool = False,
        swarm_forecast: Optional[Dict[str, Any]] = None,
        backtest_metrics: Optional[Dict[str, Any]] = None,
    ) -> ResearchReport:
        """
        Construct research report with citation verification.
        - If strict=True: any unverified quote raises CitationIntegrityError (fails the build).
        - If strict=False: unverifiable claims are dropped and counted at runtime.
        """
        verified_claims: List[Claim] = []
        dropped_claims: List[Dict[str, str]] = []

        for claim in claims:
            if self.verify_claim(claim):
                verified_claims.append(claim)
            else:
                reason = (
                    f"Quote not found verbatim in source '{claim.source_id}'"
                    if claim.source_id in self.source_filings
                    else f"Source filing '{claim.source_id}' not registered"
                )
                if strict:
                    raise CitationIntegrityError(
                        f"Citation integrity failure for {ticker}: Claim quote '{claim.quote}' "
                        f"is not a verbatim substring of source {claim.source_id}. Reason: {reason}"
                    )
                dropped_claims.append({
                    "text": claim.text,
                    "quote": claim.quote,
                    "source_id": claim.source_id,
                    "reason": reason,
                })

        # Generate markdown report
        now_utc = datetime.now(timezone.utc)
        lines: List[str] = [
            f"# ALBUS Quantitative Research Report: {ticker}",
            f"**Title**: {title}  ",
            f"**Generated**: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
            f"**Audit Status**: Verified Citations Gate: {len(verified_claims)} passed, {len(dropped_claims)} dropped  ",
            "",
            "---",
            "",
            "## 1. Verified Key Risk Disclosures & Evidence",
            "",
        ]

        if not verified_claims:
            lines.append("*No verified claims passed citation checks.*")
        else:
            for idx, c in enumerate(verified_claims, 1):
                lines.extend([
                    f"### Claim {idx}: {c.text}",
                    f"> \"{c.quote}\"",
                    f"*Source Reference: `{c.source_id}`* [VERIFIED CITATION]",
                    "",
                ])

        if swarm_forecast:
            lines.extend([
                "---",
                "## 2. Swarm Forecasting Consensus",
                f"- **Ensemble Forecast Probability**: {swarm_forecast.get('aggregated_prob', 'N/A')}",
                f"- **Brier Score / Calibration**: {swarm_forecast.get('brier_score', 'N/A')}",
                f"- **Swarm Personas Queried**: {swarm_forecast.get('personas_count', 'N/A')}",
                "",
            ])

        if backtest_metrics:
            lines.extend([
                "---",
                "## 3. Quantitative Backtest Plumbing Sanity",
                f"- **Sharpe Ratio**: {backtest_metrics.get('sharpe', 'N/A')}",
                f"- **Max Drawdown**: {backtest_metrics.get('max_drawdown', 'N/A')}",
                f"- **Turnover**: {backtest_metrics.get('turnover', 'N/A')}",
                "",
            ])

        if dropped_claims:
            lines.extend([
                "---",
                "## 4. Citation Integrity Audit (Dropped Unverified Claims)",
                f"*At runtime, {len(dropped_claims)} claims failed verbatim verification and were purged:*",
                "",
            ])
            for d in dropped_claims:
                lines.append(f"- **Dropped**: \"{d['text']}\" (Source: `{d['source_id']}`) - *{d['reason']}*")
            lines.append("")

        lines.extend([
            "---",
            "**Disclaimer**: *Research prototype for algorithmic quant evaluation. Not investment advice.*",
        ])

        markdown_content = "\n".join(lines)

        return ResearchReport(
            ticker=ticker,
            title=title,
            generated_at=now_utc,
            verified_claims=verified_claims,
            dropped_claims_count=len(dropped_claims),
            dropped_claims=dropped_claims,
            markdown_content=markdown_content,
            swarm_forecast=swarm_forecast,
            backtest_metrics=backtest_metrics,
        )
