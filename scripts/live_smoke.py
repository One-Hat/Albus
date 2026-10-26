"""
OBSERVE-ONLY LIVE SMOKE TEST -- DO NOT RUN WITHOUT LIVE NEBIUS API KEY.
Probes Nebius Token Factory endpoints for Nemotron Nano, Super, and Ultra.
Verifies response shape, reasoning fields, latency, tokens, and cost accounting.
This script performs NO mutations, NO database writes, and NO trading.
"""
from __future__ import annotations
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.llm.client import LLMClient, ModelConfigManager


PROBES: Dict[str, str] = {
    "nano": "Respond with pure JSON: {\"status\": \"ok\", \"role\": \"nano\"}",
    "super": "In one concise sentence, what is the mathematical definition of the Sharpe ratio?",
    "ultra": "Audit this claim in one sentence: 'A backtest with 2.5 Sharpe and 0% drawdown is guaranteed free of lookahead bias.'",
}


async def run_live_smoke() -> None:
    api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] Neither NEBIUS_API_KEY, GROQ_API_KEY, nor OPENROUTER_API_KEY is set.")
        print("Per AGENTS.md Rule 7, secrets must be read from environment.")
        print("Cannot run live smoke test without a live API key in environment or .env file.")
        sys.exit(1)

    provider = "Nebius" if os.environ.get("NEBIUS_API_KEY") else ("Groq" if os.environ.get("GROQ_API_KEY") else "OpenRouter")
    print("=" * 70)
    print(f"ALBUS -- LIVE {provider.upper()} REASONING SMOKE TEST (OBSERVE-ONLY)")
    print("=" * 70)

    mgr = ModelConfigManager("config/models.yaml")
    client = LLMClient(config_path="config/models.yaml")

    print(f"Base URL: {client.config_manager.endpoints.get('base_url')}")
    print(f"Concurrency Limit: {client.concurrency_limit}")
    print("-" * 70)

    results: List[Dict[str, Any]] = []

    for role in ["nano", "super", "ultra"]:
        model_cfg = mgr.get_model(role)
        prompt = PROBES[role]
        print(f"\n[PROBING {role.upper()}] Model: {model_cfg.id}")
        print(f"  Prompt: {prompt}")

        start_t = time.perf_counter()
        try:
            resp = await client.generate(
                role=role,
                messages=[{"role": "user", "content": prompt}],
                use_cache=False,
            )
            elapsed = (time.perf_counter() - start_t) * 1000.0

            has_reasoning = resp.reasoning is not None and len(resp.reasoning) > 0
            reasoning_len = len(resp.reasoning) if resp.reasoning else 0

            print(f"  [SUCCESS] HTTP 200 OK ({elapsed:.1f}ms)")
            print(f"  Tokens: Prompt={resp.prompt_tokens}, Completion={resp.completion_tokens}")
            print(f"  Cost Estimate: ${resp.cost_estimate_usd:.6f}")
            print(f"  Reasoning Captured: {has_reasoning} ({reasoning_len} chars)")
            print(f"  Content Preview: {resp.content[:150]}...")
            if has_reasoning:
                print(f"  Reasoning Preview: {resp.reasoning[:150]}...")

            results.append({
                "role": role,
                "model": model_cfg.id,
                "status": "PASS",
                "latency_ms": elapsed,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "cost_usd": resp.cost_estimate_usd,
                "has_reasoning": has_reasoning,
            })
        except Exception as exc:
            elapsed = (time.perf_counter() - start_t) * 1000.0
            print(f"  [FAIL] Error probing {role}: {exc} ({elapsed:.1f}ms)")
            results.append({
                "role": role,
                "model": model_cfg.id,
                "status": "FAIL",
                "error": str(exc),
                "latency_ms": elapsed,
            })

    print("\n" + "=" * 70)
    print("LIVE SMOKE TEST AUDIT SUMMARY")
    print("=" * 70)
    all_passed = all(r.get("status") == "PASS" for r in results)
    for r in results:
        status = r["status"]
        if status == "PASS":
            print(
                f"{r['role'].upper():<6} | {r['model']:<36} | {status} | "
                f"{r['latency_ms']:>6.1f}ms | Reasoning: {r['has_reasoning']} | ${r['cost_usd']:.6f}"
            )
        else:
            print(f"{r['role'].upper():<6} | {r['model']:<36} | {status} | Error: {r.get('error')}")
    print("=" * 70)

    if not all_passed:
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(run_live_smoke())
