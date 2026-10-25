"""
Adversarial Critic powered by Nemotron Ultra (Phase 8).
Executes deterministic checks first (lookahead, k variants, transaction costs,
sample size, forecast kinds separation), then prompts Nemotron Ultra to generate
an adversarial audit narrative and final verdict ('pass', 'weak', 'reject').
"""
from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from backtest.engine import BacktestResult
from critic.checks import CheckResult, run_deterministic_checks
from llm.client import LLMClient

logger = logging.getLogger(__name__)


class CriticVerdict(BaseModel):
    """Final audit verdict emitted by the Critic."""
    verdict: Literal["pass", "weak", "reject"]
    checks: List[CheckResult]
    narrative: str


class RawCriticLLMOutput(BaseModel):
    """Schema expected from Nemotron Ultra before rule-based verification."""
    verdict: Literal["pass", "weak", "reject"]
    narrative: str = Field(min_length=10)


class UltraCritic:
    """
    Skeptical Critic agent running on Nemotron Ultra.
    Evaluates research backtest results and is explicitly empowered to reject flawed findings.
    """

    def __init__(self, llm_client: LLMClient, role: str = "ultra"):
        self.llm_client = llm_client
        self.role = role

    def _build_audit_prompt(
        self,
        hypothesis: str,
        backtest_result: BacktestResult,
        checks: List[CheckResult],
        variants_k: int,
    ) -> List[Dict[str, str]]:
        checks_summary = [
            f"- [{ 'PASS' if c.passed else 'FAIL' }] {c.name}: {c.evidence}"
            for c in checks
        ]
        system_prompt = (
            "You are a Skeptical Senior Quantitative Risk Auditor (Nemotron Ultra). Your role is to "
            "adversarially scrutinize quantitative backtest results, challenge assumptions, and poke "
            "holes in proposed investment hypotheses. You are explicitly authorized and expected to REJECT "
            "any study suffering from lookahead bias, missing transaction costs, data mining / p-hacking, "
            "or statistically inadequate sample sizes.\n\n"
            "Verdict options:\n"
            "- 'reject': Fatal methodological error (e.g., lookahead, no costs, mixed forecast types) or severe overfitting.\n"
            "- 'weak': Technically valid but weak statistical evidence, high turnover drag, or borderline sample size.\n"
            "- 'pass': Methodologically rigorous, lookahead-free, cost-adjusted, and statistically robust.\n\n"
            f"Respond strictly in valid JSON matching this schema:\n"
            f"{json.dumps(RawCriticLLMOutput.model_json_schema(), indent=2)}"
        )
        user_prompt = (
            f"Research Hypothesis: {hypothesis}\n"
            f"Signal Variants Tested (k): {variants_k}\n\n"
            f"Backtest Performance Metrics:\n"
            f"- Sharpe Ratio: {backtest_result.sharpe:.3f} (95% CI: [{backtest_result.sharpe_ci_95[0]:.3f}, {backtest_result.sharpe_ci_95[1]:.3f}])\n"
            f"- Max Drawdown: {backtest_result.max_drawdown:.2%}\n"
            f"- Turnover: {backtest_result.turnover:.3f}\n"
            f"- Total Cost Drag: {backtest_result.total_cost_drag:.4f}\n"
            f"- Total Return: {backtest_result.total_return:.2%}\n"
            f"- Trade Sample Size: {backtest_result.sample_size}\n\n"
            f"Deterministic Code Audit Findings:\n"
            + "\n".join(checks_summary)
            + "\n\nProvide your skeptical adversarial critique narrative and final verdict."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def audit(
        self,
        hypothesis: str,
        backtest_result: BacktestResult,
        variants_k: int = 1,
        kind_separation_verified: bool = True,
        min_trades: int = 30,
        max_allowed_k: int = 15,
    ) -> CriticVerdict:
        """
        Execute deterministic checks, invoke Nemotron Ultra for adversarial synthesis,
        and enforce hard failure overrides.
        """
        # 1. Run deterministic checks first
        checks = run_deterministic_checks(
            backtest_result=backtest_result,
            variants_k=variants_k,
            kind_separation_verified=kind_separation_verified,
            min_trades=min_trades,
            max_allowed_k=max_allowed_k,
        )

        fatal_checks = {"signal_vs_filing_date", "transaction_costs_included", "forecast_kinds_separated"}
        warning_checks = {"sample_size_adequate", "k_variants_multiple_testing"}

        has_fatal_failure = any(not c.passed for c in checks if c.name in fatal_checks)
        has_warning_failure = any(not c.passed for c in checks if c.name in warning_checks)

        # 2. Invoke Nemotron Ultra for narrative critique
        messages = self._build_audit_prompt(hypothesis, backtest_result, checks, variants_k)
        raw_output: RawCriticLLMOutput = await self.llm_client.generate_structured(
            role=self.role,
            messages=messages,
            response_model=RawCriticLLMOutput,
            max_repairs=2,
        )

        final_verdict = raw_output.verdict

        # 3. Hard rule overrides: Code checks strictly bound the LLM verdict
        if has_fatal_failure:
            final_verdict = "reject"
        elif has_warning_failure and final_verdict == "pass":
            final_verdict = "weak"

        return CriticVerdict(
            verdict=final_verdict,
            checks=checks,
            narrative=raw_output.narrative,
        )
