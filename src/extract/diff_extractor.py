"""
Filing-diff extractor powered by Nemotron Nano.
Compares year t against year t-1 per section and returns FilingDiff JSON
through a validate-and-repair loop with strict verbatim quote checking.
"""
from __future__ import annotations
import json
import logging
from typing import Any, Dict, List
from pydantic import BaseModel, Field, ValidationError

from data.sections import FilingSection
from extract.quote_check import (
    Claim,
    FilingDiff,
    find_unverified_quotes,
    verify_and_filter_diff,
)
from llm.client import LLMClient

logger = logging.getLogger(__name__)


class RawDiffOutput(BaseModel):
    """Schema requested from the LLM before attaching filing metadata."""
    new_risks: List[Claim] = Field(default_factory=list)
    removed_risks: List[Claim] = Field(default_factory=list)
    tone_shift: float = Field(ge=-1.0, le=1.0)


class DiffExtractor:
    """
    Extracts structured risk factor diffs between year t and year t-1 filings
    using Nemotron Nano with a Pydantic + verbatim quote validate-and-repair loop.
    """

    def __init__(self, llm_client: LLMClient, role: str = "nano"):
        self.llm_client = llm_client
        self.role = role

    def _build_messages(
        self,
        section_t: FilingSection,
        section_t_minus_1: FilingSection,
    ) -> List[Dict[str, str]]:
        schema_str = json.dumps(RawDiffOutput.model_json_schema(), indent=2)
        system_prompt = (
            "You are a quantitative financial filing analyst. Compare the company's Item 1A Risk Factors "
            "from year t against year t-1. Identify newly added risk claims (new_risks), removed risk claims "
            "(removed_risks), and estimate the overall tone_shift (-1.0 = much more negative/risky, "
            "0.0 = neutral, +1.0 = much more positive/confident).\n"
            "CRITICAL RULE: Every `quote` in `new_risks` MUST be an exact verbatim substring copied "
            "character-for-character from the Year t text. Every `quote` in `removed_risks` MUST be an "
            "exact verbatim substring copied character-for-character from the Year t-1 text.\n"
            f"Respond ONLY with valid JSON matching this schema:\n{schema_str}"
        )
        user_prompt = (
            f"Ticker: {section_t.ticker}\n\n"
            f"--- YEAR t ({section_t.accession_number}) ---\n"
            f"{section_t.text}\n\n"
            f"--- YEAR t-1 ({section_t_minus_1.accession_number}) ---\n"
            f"{section_t_minus_1.text}\n"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _strip_code_fences(raw_text: str) -> str:
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned

    async def extract_diff(
        self,
        section_t: FilingSection,
        section_t_minus_1: FilingSection,
        max_repairs: int = 2,
    ) -> FilingDiff:
        """
        Compare year t against year t-1 and return a verified FilingDiff.
        """
        if section_t.ticker != section_t_minus_1.ticker:
            raise ValueError(
                f"Ticker mismatch between year t ({section_t.ticker}) and year t-1 ({section_t_minus_1.ticker})"
            )

        messages = self._build_messages(section_t, section_t_minus_1)
        last_valid_raw: RawDiffOutput | None = None

        for attempt in range(max_repairs + 1):
            response = await self.llm_client.generate(role=self.role, messages=messages)
            cleaned_json = self._strip_code_fences(response.content)

            try:
                parsed_dict = json.loads(cleaned_json)
                raw_diff = RawDiffOutput.model_validate(parsed_dict)
                last_valid_raw = raw_diff
            except (json.JSONDecodeError, ValidationError) as err:
                if attempt >= max_repairs:
                    raise RuntimeError(
                        f"DiffExtractor failed Pydantic validation after {max_repairs} repairs "
                        f"for {section_t.ticker}: {err}"
                    ) from err

                logger.warning(
                    f"DiffExtractor JSON/schema failure on attempt {attempt + 1} for {section_t.ticker}: {err}"
                )
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your output failed JSON/Pydantic validation:\n{err}\n"
                        "Return ONLY a valid JSON object matching the schema."
                    ),
                })
                continue

            # Check verbatim quotes
            quote_errors = find_unverified_quotes(
                raw_diff.new_risks,
                raw_diff.removed_risks,
                section_t.text,
                section_t_minus_1.text,
            )
            if not quote_errors:
                # All quotes verified verbatim!
                break

            if attempt < max_repairs:
                err_msg = "\n".join(quote_errors)
                logger.warning(
                    f"DiffExtractor quote verification failed on attempt {attempt + 1} "
                    f"for {section_t.ticker}:\n{err_msg}"
                )
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": (
                        "One or more quotes failed verbatim substring verification:\n"
                        f"{err_msg}\n"
                        "Replace every invalid quote with an EXACT verbatim substring from the respective filing text, "
                        "or remove the unverified claim."
                    ),
                })

        if last_valid_raw is None:
            raise RuntimeError(f"DiffExtractor produced no valid output for {section_t.ticker}")

        candidate_diff = FilingDiff(
            ticker=section_t.ticker,
            accepted_at=section_t.accepted_at,
            new_risks=last_valid_raw.new_risks,
            removed_risks=last_valid_raw.removed_risks,
            tone_shift=last_valid_raw.tone_shift,
        )

        # Enforce strict rejection of any remaining unverified quotes
        verified_diff, _ = verify_and_filter_diff(
            candidate_diff,
            section_t.text,
            section_t_minus_1.text,
        )
        return verified_diff
