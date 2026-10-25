"""
Forecasting Swarm Personas and Runner:
8 diverse personas on Nemotron Nano and 3 deep-reasoning personas on Nemotron Super.
Each agent emits p_yes (clipped/validated in [0.02, 0.98]), rationale, and evidence_ids,
which are appended to the hash-chained ForecastLedger.
"""
from __future__ import annotations
import json
from datetime import datetime
from typing import List, Literal, Optional
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field

from extract.quote_check import FilingDiff
from llm.client import LLMClient
from swarm.ledger import Forecast, ForecastLedger
from swarm.questions import Question

EASTERN = ZoneInfo("America/New_York")


class PersonaSpec(BaseModel):
    agent_id: str
    name: str
    role: Literal["nano", "super"]
    system_prompt: str


class SwarmAgentOutput(BaseModel):
    p_yes: float = Field(ge=0.02, le=0.98)
    rationale: str = Field(min_length=1)
    evidence_ids: List[str] = Field(default_factory=list)


NANO_PERSONAS: List[PersonaSpec] = [
    PersonaSpec(
        agent_id="nano_skeptic",
        name="Skeptic",
        role="nano",
        system_prompt=(
            "You are a skeptical quant forecaster. Assume most new 10-K risk factor language is "
            "defensive legal boilerplate unless it cites specific, concrete events or financial impacts."
        ),
    ),
    PersonaSpec(
        agent_id="nano_base_rate_anchorer",
        name="Base-Rate Anchorer",
        role="nano",
        system_prompt=(
            "You anchor strongly on the historical base rate (0.50 for excess return underperformance) "
            "and only deviate modestly when multiple independent risk additions point in the same direction."
        ),
    ),
    PersonaSpec(
        agent_id="nano_risk_language_reader",
        name="Risk-Language Reader",
        role="nano",
        system_prompt=(
            "You closely analyze shifts from hypothetical phrasing ('may adversely affect') to "
            "realized phrasing ('has adversely affected' or 'we are currently subject to')."
        ),
    ),
    PersonaSpec(
        agent_id="nano_tone_shift_specialist",
        name="Tone-Shift Specialist",
        role="nano",
        system_prompt=(
            "You weight the net balance between newly added risks, removed risks, and the overall "
            "tone_shift score across Item 1A."
        ),
    ),
    PersonaSpec(
        agent_id="nano_litigation_auditor",
        name="Litigation & Regulatory Auditor",
        role="nano",
        system_prompt=(
            "You focus on newly disclosed lawsuits, regulatory investigations, subpoenas, and "
            "compliance enforcement actions."
        ),
    ),
    PersonaSpec(
        agent_id="nano_supply_chain_analyst",
        name="Supply Chain & Operations Analyst",
        role="nano",
        system_prompt=(
            "You focus on manufacturing bottlenecks, single-source foundry dependencies, export controls, "
            "and tariff exposures."
        ),
    ),
    PersonaSpec(
        agent_id="nano_macro_credit_watcher",
        name="Macro & Credit Watcher",
        role="nano",
        system_prompt=(
            "You focus on debt covenants, refinancing risks, inflation, FX headwinds, and enterprise "
            "demand slowdowns."
        ),
    ),
    PersonaSpec(
        agent_id="nano_tail_risk_hunter",
        name="Tail-Risk Hunter",
        role="nano",
        system_prompt=(
            "You look for asymmetric, high-severity tail risks (cyber breaches, key product cancellations, "
            "or major customer concentration losses)."
        ),
    ),
]

SUPER_PERSONAS: List[PersonaSpec] = [
    PersonaSpec(
        agent_id="super_historical_synthesizer",
        name="Historical Synthesizer",
        role="super",
        system_prompt=(
            "You are a senior quantitative strategist synthesizing SEC Item 1A filing diffs with "
            "historical empirical base rates and cross-sectional factor behavior."
        ),
    ),
    PersonaSpec(
        agent_id="super_bayesian_updater",
        name="Bayesian Updater",
        role="super",
        system_prompt=(
            "You explicitly start from a 50% prior and apply calibrated likelihood ratios for each "
            "verified new or removed risk claim to arrive at a posterior probability in [0.02, 0.98]."
        ),
    ),
    PersonaSpec(
        agent_id="super_cross_sectional_ranker",
        name="Cross-Sectional Ranker",
        role="super",
        system_prompt=(
            "You evaluate whether the company's newly added risk disclosures represent idiosyncratic "
            "deterioration relative to its sector ETF benchmark."
        ),
    ),
]

ALL_PERSONAS: List[PersonaSpec] = NANO_PERSONAS + SUPER_PERSONAS


class SwarmRunner:
    """
    Executes all swarm personas on a Question + FilingDiff and records
    their hash-chained forecasts into the ForecastLedger.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        ledger: ForecastLedger,
        personas: Optional[List[PersonaSpec]] = None,
    ):
        self.llm_client = llm_client
        self.ledger = ledger
        self.personas = personas if personas is not None else ALL_PERSONAS

    def _build_prompt(self, persona: PersonaSpec, question: Question, diff: FilingDiff) -> List[dict]:
        diff_summary = {
            "ticker": diff.ticker,
            "accepted_at": diff.accepted_at.isoformat(),
            "tone_shift": diff.tone_shift,
            "new_risks": [c.model_dump() for c in diff.new_risks],
            "removed_risks": [c.model_dump() for c in diff.removed_risks],
        }
        user_msg = (
            f"Question ID: {question.question_id}\n"
            f"Ticker: {question.ticker}\n"
            f"Event Definition: {question.event_def}\n"
            f"Horizon (trading days): {question.horizon_days}\n\n"
            f"Filing Diff Evidence:\n{json.dumps(diff_summary, indent=2)}\n\n"
            "Output your probability `p_yes` (strictly between 0.02 and 0.98), a concise `rationale`, "
            "and `evidence_ids` referencing the `source_id` of any claims you relied upon."
        )
        return [
            {"role": "system", "content": persona.system_prompt},
            {"role": "user", "content": user_msg},
        ]

    async def forecast_question(
        self,
        question: Question,
        diff: FilingDiff,
        made_at: Optional[datetime] = None,
    ) -> List[Forecast]:
        timestamp = (made_at or datetime.now(EASTERN)).astimezone(EASTERN)
        recorded_forecasts: List[Forecast] = []

        for persona in self.personas:
            model_cfg = self.llm_client.config_manager.get_model(persona.role)
            messages = self._build_prompt(persona, question, diff)

            agent_out: SwarmAgentOutput = await self.llm_client.generate_structured(
                role=persona.role,
                messages=messages,
                response_model=SwarmAgentOutput,
                max_repairs=2,
            )

            # Clip to [0.02, 0.98] as an extra safeguard while SwarmAgentOutput already enforces [0.02, 0.98]
            clipped_p = min(0.98, max(0.02, float(agent_out.p_yes)))

            forecast = self.ledger.append_forecast(
                question_id=question.question_id,
                agent_id=persona.agent_id,
                model=model_cfg.id,
                p_yes=clipped_p,
                rationale=agent_out.rationale,
                evidence_ids=agent_out.evidence_ids,
                made_at=timestamp,
            )
            recorded_forecasts.append(forecast)

        return recorded_forecasts
