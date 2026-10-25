"""
Autonomous Research Agent powered by Nemotron Super.
Runs a tool-calling loop with run_backtest, get_stats, plot, and list_filings,
enforces max-iterations caps, and tracks the counter of signal variants tested (k).
"""
from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError

from backtest.engine import BacktestResult
from llm.client import LLMClient
from research.tools import ResearchTools, ToolCallRequest

logger = logging.getLogger(__name__)


class ResearchReport(BaseModel):
    """Structured report produced by the research agent."""
    hypothesis: str
    variants_tested_k: int
    iterations: int
    final_backtest: Optional[BacktestResult] = None
    summary_narrative: str
    trace: List[Dict[str, Any]] = Field(default_factory=list)


class ResearchAgent:
    """
    Research agent manager (Nemotron Super).
    Takes a natural language research hypothesis, iteratively invokes quant tools,
    evaluates multiple signal variants (tracking k), and produces a structured research report.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: ResearchTools,
        role: str = "super",
        max_iterations: int = 10,
    ):
        self.llm_client = llm_client
        self.tools = tools
        self.role = role
        self.max_iterations = max_iterations

    def _build_system_prompt(self) -> str:
        tool_desc = self.tools.get_tool_descriptions()
        schema_desc = json.dumps(ToolCallRequest.model_json_schema(), indent=2)
        return (
            "You are a Senior Quantitative Research Manager (Nemotron Super). Your goal is to "
            "rigorously test and validate investment hypotheses using historical SEC filings and backtest tools.\n\n"
            f"{tool_desc}\n\n"
            "INSTRUCTIONS:\n"
            "- First explore filings or test signal variants using run_backtest.\n"
            "- Inspect distributional stats and chart data using get_stats and plot.\n"
            "- When you have gathered sufficient evidence or compared variants, finish by issuing 'final_answer'.\n"
            "- You must respond strictly in JSON matching this schema:\n"
            f"{schema_desc}"
        )

    @staticmethod
    def _strip_fences(text: str) -> str:
        s = text.strip()
        if s.startswith("```json"):
            s = s[len("```json"):].strip()
        elif s.startswith("```"):
            s = s[len("```"):].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
        return s

    async def run_hypothesis(self, hypothesis: str) -> ResearchReport:
        """
        Execute the autonomous research loop on the given hypothesis.
        Stops cleanly when the agent issues 'final_answer' or when max_iterations is reached.
        """
        system_prompt = self._build_system_prompt()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research Hypothesis: {hypothesis}"},
        ]
        trace: List[Dict[str, Any]] = []
        final_summary: Optional[str] = None
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            response = await self.llm_client.generate(role=self.role, messages=messages)
            raw_content = self._strip_fences(response.content)

            try:
                parsed = json.loads(raw_content)
                tool_req = ToolCallRequest.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as err:
                logger.warning(f"Research agent emitted unparseable action on iteration {iteration}: {err}")
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": f"Output failed schema validation: {err}. Respond strictly with valid JSON.",
                })
                continue

            trace.append({
                "iteration": iteration,
                "role": self.role,
                "action": tool_req.action,
                "tool_name": tool_req.tool_name,
                "arguments": tool_req.arguments,
                "reasoning": response.reasoning,
            })

            if tool_req.action == "final_answer":
                final_summary = tool_req.summary or response.content
                break

            if tool_req.action == "call_tool" and tool_req.tool_name:
                tool_output = self.tools.execute_tool(tool_req.tool_name, tool_req.arguments)
                trace[-1]["tool_output"] = tool_output

                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": f"Tool '{tool_req.tool_name}' returned:\n{json.dumps(tool_output, default=str)}",
                })

        if final_summary is None:
            # Hit iteration cap without final answer
            final_summary = (
                f"Agent hit maximum iteration cap ({self.max_iterations}). "
                f"Explored {self.tools.k} signal variants. Last backtest recorded."
            )
            trace.append({
                "iteration": iteration,
                "role": "system",
                "action": "cap_reached",
                "summary": final_summary,
            })

        return ResearchReport(
            hypothesis=hypothesis,
            variants_tested_k=self.tools.k,
            iterations=iteration,
            final_backtest=self.tools.last_backtest_result,
            summary_narrative=final_summary,
            trace=trace,
        )
