"""
MockLLM provider for deterministic offline testing.
AGENTS.md Rule 4: MockLLM may only be imported under tests/. Never in src/.
"""
from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional


class MockLLM:
    """
    Mock LLM provider designed for unit testing without live API keys or network.
    Emits configurable completions, reasoning fields, token counts, and latency.
    """

    def __init__(self, delay_ms: float = 0.0):
        self.delay_ms = delay_ms
        self.registered_responses: Dict[str, Dict[str, Any]] = {}
        self.registered_sequences: Dict[str, List[Dict[str, Any]]] = {}
        self.call_counts: Dict[str, int] = {}
        self.failing_sequences: Dict[str, Dict[str, Any]] = {}
        self.active_calls: int = 0
        self.max_active_calls: int = 0

    def register_response(
        self,
        role_or_model: str,
        content: str,
        reasoning: Optional[str] = None,
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
    ) -> None:
        """Register a single static response for a model or role."""
        self.registered_responses[role_or_model] = {
            "content": content,
            "reasoning": reasoning or f"Mock reasoning for {role_or_model}",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def register_sequence(
        self,
        role_or_model: str,
        responses: List[Dict[str, Any]],
    ) -> None:
        """Register a sequence of responses to be returned in order."""
        self.registered_sequences[role_or_model] = list(responses)

    def register_failing_sequence(
        self,
        role_or_model: str,
        fail_count: int,
        success_content: str,
        success_reasoning: str = "Reasoning after recovery",
    ) -> None:
        """Register a sequence that fails fail_count times before succeeding."""
        self.failing_sequences[role_or_model] = {
            "fail_count": fail_count,
            "success_content": success_content,
            "success_reasoning": success_reasoning,
            "current_fails": 0,
        }

    async def generate(
        self,
        model_id: str,
        role: str,
        messages: List[Dict[str, str]],
        **params: Any,
    ) -> Dict[str, Any]:
        """Simulate LLM generation."""
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.delay_ms > 0:
                await asyncio.sleep(self.delay_ms / 1000.0)

            lookup_key = role if role in self.registered_responses or role in self.registered_sequences or role in self.failing_sequences else model_id

            self.call_counts[lookup_key] = self.call_counts.get(lookup_key, 0) + 1

            # Check failing sequence
            if lookup_key in self.failing_sequences:
                fstate = self.failing_sequences[lookup_key]
                if fstate["current_fails"] < fstate["fail_count"]:
                    fstate["current_fails"] += 1
                    raise RuntimeError(f"Simulated rate limit error 429 for {lookup_key}")
                return {
                    "content": fstate["success_content"],
                    "reasoning": fstate["success_reasoning"],
                    "prompt_tokens": 80,
                    "completion_tokens": 40,
                    "model": model_id,
                }

            # Check sequence
            if lookup_key in self.registered_sequences and self.registered_sequences[lookup_key]:
                resp = self.registered_sequences[lookup_key].pop(0)
                return {
                    "content": resp.get("content", ""),
                    "reasoning": resp.get("reasoning", "Mock sequential reasoning"),
                    "prompt_tokens": resp.get("prompt_tokens", 100),
                    "completion_tokens": resp.get("completion_tokens", 50),
                    "model": model_id,
                }

            # Check static response
            if lookup_key in self.registered_responses:
                resp = self.registered_responses[lookup_key]
                return {
                    "content": resp["content"],
                    "reasoning": resp["reasoning"],
                    "prompt_tokens": resp["prompt_tokens"],
                    "completion_tokens": resp["completion_tokens"],
                    "model": model_id,
                }

            # Default fallback response
            return {
                "content": f"Default mock response for {lookup_key}",
                "reasoning": f"Default mock reasoning for {lookup_key}",
                "prompt_tokens": 50,
                "completion_tokens": 25,
                "model": model_id,
            }
        finally:
            self.active_calls -= 1
