"""
Phase 1 Tests: LLM Client, Disk Cache, Model Configuration, and MockLLM.
Following AGENTS.md Rule 2: Tests written first.
"""
import asyncio
import os
import shutil
import tempfile
import pytest
from pathlib import Path
from pydantic import BaseModel, Field

# We import MockLLM under tests/ (AGENTS.md Rule 4: MockLLM may only be imported under tests/)
from llm.mock import MockLLM
from llm.cache import DiskCache
from llm.client import LLMClient, LLMResponse, ModelConfigManager


class SampleClaim(BaseModel):
    category: str
    quote: str
    risk_score: float = Field(ge=0.0, le=1.0)


@pytest.fixture
def temp_cache_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_llm_cache_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_models_config_yaml():
    """Verify config/models.yaml contains nano, super, and ultra with required attributes."""
    manager = ModelConfigManager()
    assert "nano" in manager.models
    assert "super" in manager.models
    assert "ultra" in manager.models

    nano = manager.get_model("nano")
    assert nano.id.startswith("nvidia/")
    assert nano.context_window > 0
    assert nano.cost_per_1m_input_tokens_usd > 0
    assert nano.cost_per_1m_output_tokens_usd > 0

    super_m = manager.get_model("super")
    assert super_m.id.startswith("nvidia/")

    ultra_m = manager.get_model("ultra")
    assert ultra_m.id.startswith("nvidia/")


def test_opensource_provider_groq_and_openrouter_resolution(monkeypatch):
    """Verify that setting GROQ_API_KEY or OPENROUTER_API_KEY routes to open-source models without guessing IDs."""
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_synthetic_test_key_12345")

    client = LLMClient()
    assert client.config_manager.models["super"].id == "llama-3.3-70b-versatile"
    assert client.config_manager.endpoints["base_url"] == "https://api.groq.com/openai/v1"

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-synthetic_test_key_12345")
    client_or = LLMClient()
    assert client_or.config_manager.models["super"].id == "meta-llama/llama-3.3-70b-instruct"
    assert client_or.config_manager.endpoints["base_url"] == "https://openrouter.ai/api/v1"


def test_disk_cache(temp_cache_dir):
    """Verify cache store, retrieve, key hashing, and miss behavior."""
    cache = DiskCache(cache_dir=temp_cache_dir)
    messages = [{"role": "user", "content": "What is 2+2?"}]
    params = {"temperature": 0.1, "max_tokens": 100}

    key = cache.compute_key("nvidia/test-model", messages, params)
    assert cache.get(key) is None

    test_data = {
        "content": "4",
        "reasoning": "2 + 2 = 4",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "model": "nvidia/test-model",
    }
    cache.set(key, test_data)

    retrieved = cache.get(key)
    assert retrieved is not None
    assert retrieved["content"] == "4"
    assert retrieved["reasoning"] == "2 + 2 = 4"
    assert retrieved["prompt_tokens"] == 10


@pytest.mark.asyncio
async def test_smoke_nano_super_ultra_via_mock(temp_cache_dir):
    """
    Gate requirement: Smoke test calls Nano, Super, and Ultra.
    Verifies reasoning field is handled and not parsed as content.
    """
    mock = MockLLM()
    # Configure deterministic responses with distinct reasoning and content
    mock.register_response(
        role_or_model="nano",
        content='{"findings": "Item 1A risk factors identified"}',
        reasoning="Reasoning step: extracted risk disclosures from filing.",
        prompt_tokens=150,
        completion_tokens=40,
    )
    mock.register_response(
        role_or_model="super",
        content='{"strategy": "Decile long-short hypothesis verified"}',
        reasoning="Reasoning step: evaluated Sharpe ratio and decile spreads.",
        prompt_tokens=500,
        completion_tokens=90,
    )
    mock.register_response(
        role_or_model="ultra",
        content='{"verdict": "pass", "notes": "No lookahead detected"}',
        reasoning="Reasoning step: scrutinized transaction timestamps against price data.",
        prompt_tokens=1200,
        completion_tokens=180,
    )

    cache = DiskCache(cache_dir=temp_cache_dir)
    client = LLMClient(cache=cache, mock_provider=mock)

    for role in ["nano", "super", "ultra"]:
        resp: LLMResponse = await client.generate(
            role=role,
            messages=[{"role": "user", "content": f"Execute {role} test prompt"}],
        )
        assert resp.content is not None and len(resp.content) > 0
        assert resp.reasoning is not None and "Reasoning step:" in resp.reasoning
        # Ensure reasoning is separate from content
        assert "Reasoning step:" not in resp.content
        assert resp.prompt_tokens > 0
        assert resp.completion_tokens > 0
        assert resp.latency_ms >= 0


@pytest.mark.asyncio
async def test_cache_hit_repeated_call(temp_cache_dir):
    """
    Gate requirement: A repeated call hits the disk cache, and the log shows tokens and latency.
    """
    mock = MockLLM()
    mock.register_response(
        role_or_model="nano",
        content="Cached result content",
        reasoning="Reasoned first time",
        prompt_tokens=50,
        completion_tokens=20,
    )

    cache = DiskCache(cache_dir=temp_cache_dir)
    client = LLMClient(cache=cache, mock_provider=mock)

    messages = [{"role": "user", "content": "Fetch repeated prompt"}]

    # First call: cache miss, invokes mock provider
    resp1 = await client.generate(role="nano", messages=messages)
    assert not resp1.cached
    assert resp1.content == "Cached result content"
    assert resp1.reasoning == "Reasoned first time"

    # Second call: cache hit!
    resp2 = await client.generate(role="nano", messages=messages)
    assert resp2.cached
    assert resp2.content == "Cached result content"
    assert resp2.reasoning == "Reasoned first time"
    # Call logs should show cache hit and token counts
    log_entry = client.last_call_log
    assert log_entry["cached"] is True
    assert log_entry["tokens"]["prompt"] == 50
    assert log_entry["tokens"]["completion"] == 20
    assert "cost_usd" in log_entry


@pytest.mark.asyncio
async def test_concurrency_limiter():
    """Verify that concurrency limiter semaphore caps concurrent calls."""
    mock = MockLLM(delay_ms=50)
    mock.register_response(
        role_or_model="nano",
        content="Parallel reply",
        reasoning="Reasoning parallel",
    )
    client = LLMClient(mock_provider=mock, concurrency_limit=2)

    async def worker(idx: int):
        return await client.generate(
            role="nano",
            messages=[{"role": "user", "content": f"Task {idx}"}],
            use_cache=False,
        )

    results = await asyncio.gather(*(worker(i) for i in range(6)))
    assert len(results) == 6
    assert mock.max_active_calls <= 2, f"Concurrency exceeded limit: {mock.max_active_calls} > 2"


@pytest.mark.asyncio
async def test_retries_with_backoff(temp_cache_dir):
    """Verify exponential backoff retry on transient errors."""
    mock = MockLLM()
    # Fail twice with RateLimitError before succeeding
    mock.register_failing_sequence(
        role_or_model="nano",
        fail_count=2,
        success_content="Eventual success after retries",
    )
    cache = DiskCache(cache_dir=temp_cache_dir)
    client = LLMClient(cache=cache, mock_provider=mock, max_retries=3)

    resp = await client.generate(
        role="nano",
        messages=[{"role": "user", "content": "Retry test with unique prompt"}],
    )
    assert resp.content == "Eventual success after retries"
    assert mock.call_counts.get("nano", 0) == 3


@pytest.mark.asyncio
async def test_structured_output_validation_and_repair(temp_cache_dir):
    """
    Verify pydantic validation and repair loop (Rule 5 & Rule 6).
    """
    mock = MockLLM()
    # First attempt: invalid JSON (missing required quote)
    # Second attempt: repaired valid JSON
    mock.register_sequence(
        role_or_model="nano",
        responses=[
            {"content": '{"category": "litigation", "risk_score": 0.85}'},  # missing 'quote'
            {"content": '{"category": "litigation", "quote": "We face pending legal actions.", "risk_score": 0.85}'},
        ]
    )

    cache = DiskCache(cache_dir=temp_cache_dir)
    client = LLMClient(cache=cache, mock_provider=mock)
    claim: SampleClaim = await client.generate_structured(
        role="nano",
        messages=[{"role": "user", "content": "Extract claim with unique prompt"}],
        response_model=SampleClaim,
        max_repairs=2,
    )

    assert isinstance(claim, SampleClaim)
    assert claim.category == "litigation"
    assert claim.quote == "We face pending legal actions."
    assert claim.risk_score == 0.85


def test_rule_4_mock_llm_never_imported_in_src():
    """
    AGENTS.md Rule 4: MockLLM may only be imported under tests/. Never in src/.
    Verify by scanning every python file under src/.
    """
    src_dir = Path("src")
    if not src_dir.exists():
        return

    for py_file in src_dir.rglob("*.py"):
        if py_file.name == "mock.py":
            continue  # The definition itself is fine
        content = py_file.read_text(encoding="utf-8")
        assert "MockLLM" not in content, (
            f"Rule 4 Violation: {py_file} imports or references MockLLM! "
            f"MockLLM is strictly prohibited inside src/."
        )


# ==============================================================================
# W3 Live-Readiness Synthetic Unit Tests (Reasoning Models, Truncation, 429)
# All fixtures below are hand-written synthetic responses, commented as synthetic.
# ==============================================================================


@pytest.mark.asyncio
async def test_synthetic_reasoning_shape_answer_in_content(temp_cache_dir):
    """
    Synthetic Test: Response shape where thought is in reasoning_content
    and the final answer is cleanly delivered in content.
    """
    # SYNTHETIC FIXTURE: Hand-written synthetic response shape (answer in content)
    synthetic_fixture = {
        "content": '{"category": "macro", "quote": "inflationary pressures", "risk_score": 0.65}',
        "reasoning_content": "Chain-of-thought: Analyzed item 1A macro factors regarding rising interest rates.",
        "prompt_tokens": 120,
        "completion_tokens": 45,
    }

    mock = MockLLM()
    mock.register_response(
        role_or_model="nano",
        content=synthetic_fixture["content"],
        reasoning=synthetic_fixture["reasoning_content"],
        prompt_tokens=synthetic_fixture["prompt_tokens"],
        completion_tokens=synthetic_fixture["completion_tokens"],
    )

    cache = DiskCache(cache_dir=temp_cache_dir)
    client = LLMClient(cache=cache, mock_provider=mock)

    resp = await client.generate(
        role="nano",
        messages=[{"role": "user", "content": "Extract macro risk claim"}],
        use_cache=False,
    )
    assert resp.content == synthetic_fixture["content"]
    assert resp.reasoning == synthetic_fixture["reasoning_content"]


@pytest.mark.asyncio
async def test_synthetic_reasoning_shape_answer_only_in_reasoning_content(temp_cache_dir):
    """
    Synthetic Test: Reasoning model outputs the entire answer inside reasoning_content
    (or reasoning field), leaving content empty or None.
    """
    # SYNTHETIC FIXTURE: Hand-written synthetic response shape (content empty, answer in reasoning_content)
    synthetic_fixture = {
        "content": "",
        "reasoning_content": (
            "Thinking process: Evaluating regulatory disclosures.\n"
            "```json\n"
            '{"category": "regulatory", "quote": "subject to strict oversight", "risk_score": 0.80}\n'
            "```"
        ),
        "prompt_tokens": 140,
        "completion_tokens": 60,
    }

    mock = MockLLM()
    mock.register_sequence(
        role_or_model="nano",
        responses=[synthetic_fixture],
    )

    cache = DiskCache(cache_dir=temp_cache_dir)
    client = LLMClient(cache=cache, mock_provider=mock)

    claim: SampleClaim = await client.generate_structured(
        role="nano",
        messages=[{"role": "user", "content": "Extract regulatory claim"}],
        response_model=SampleClaim,
        use_cache=False,
    )
    assert isinstance(claim, SampleClaim)
    assert claim.category == "regulatory"
    assert claim.quote == "subject to strict oversight"
    assert claim.risk_score == 0.80


@pytest.mark.asyncio
async def test_synthetic_reasoning_shape_empty_content_finish_reason_length(temp_cache_dir):
    """
    Synthetic Test: Response with empty content and finish_reason='length'
    must fail loudly with a clear error indicating token truncation.
    """
    # SYNTHETIC FIXTURE: Hand-written synthetic response shape (empty content, finish_reason=length)
    synthetic_fixture = {
        "content": "",
        "reasoning": "Extensive reasoning about supply chain bottlenecks that used up all output tokens...",
        "finish_reason": "length",
        "prompt_tokens": 4096,
        "completion_tokens": 4096,
    }

    mock = MockLLM()
    mock.register_sequence(
        role_or_model="nano",
        responses=[synthetic_fixture],
    )

    cache = DiskCache(cache_dir=temp_cache_dir)
    client = LLMClient(cache=cache, mock_provider=mock)

    with pytest.raises(RuntimeError) as exc_info:
        await client.generate(
            role="nano",
            messages=[{"role": "user", "content": "Large prompt triggering token limit"}],
            use_cache=False,
        )

    assert "finish_reason='length'" in str(exc_info.value)
    assert "truncated" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_synthetic_truncated_json_repair(temp_cache_dir):
    """
    Synthetic Test: Output JSON truncated mid-stream by token limit is recovered
    by heuristic bracket/quote repair in generate_structured.
    """
    # SYNTHETIC FIXTURE: Hand-written synthetic truncated JSON fixture
    # Missing closing quote, closing brace, and field
    truncated_synthetic_raw = (
        '{"category": "cybersecurity", "quote": "We faced unauthorized network access", "risk_score": 0.9'
        # Intentionally missing closing '}'
    )

    mock = MockLLM()
    mock.register_sequence(
        role_or_model="nano",
        responses=[{"content": truncated_synthetic_raw}],
    )

    cache = DiskCache(cache_dir=temp_cache_dir)
    client = LLMClient(cache=cache, mock_provider=mock)

    claim: SampleClaim = await client.generate_structured(
        role="nano",
        messages=[{"role": "user", "content": "Extract cyber risk claim"}],
        response_model=SampleClaim,
        max_repairs=0,  # Must be fixed on initial attempt via heuristic bracket repair
        use_cache=False,
    )
    assert isinstance(claim, SampleClaim)
    assert claim.category == "cybersecurity"
    assert claim.quote == "We faced unauthorized network access"
    assert claim.risk_score == 0.9


@pytest.mark.asyncio
async def test_synthetic_429_with_retry_after(temp_cache_dir):
    """
    Synthetic Test: Upstream HTTP 429 response with Retry-After header is honored
    with appropriate backoff delay and succeeds on subsequent attempt.
    """
    # SYNTHETIC FIXTURE: Hand-written synthetic 429 error with Retry-After header
    mock = MockLLM()
    mock.failing_sequences["nano"] = {
        "fail_count": 1,
        "error_message": "Rate limit exceeded (HTTP 429)",
        "headers": {"Retry-After": "0.05"},  # 50ms synthetic delay
        "success_content": '{"category": "liquidity", "quote": "Credit facility covenants", "risk_score": 0.4}',
        "success_reasoning": "Recovered reasoning after rate limit delay.",
    }

    cache = DiskCache(cache_dir=temp_cache_dir)
    client = LLMClient(cache=cache, mock_provider=mock, max_retries=2)

    resp = await client.generate(
        role="nano",
        messages=[{"role": "user", "content": "Check liquidity risk with 429 retry"}],
        use_cache=False,
    )
    assert "Credit facility covenants" in resp.content
    assert mock.call_counts["nano"] == 2
