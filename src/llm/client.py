"""
Async LLM Client for Nebius Token Factory (OpenAI-compatible API).
Supports Nemotron Nano, Super, Ultra with reasoning fields, concurrency limits,
exponential backoff, disk caching, and Pydantic structured output validation.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar
import yaml
from pydantic import BaseModel, ValidationError

try:
    from openai import AsyncOpenAI, APIConnectionError, RateLimitError, InternalServerError, APIStatusError
except ImportError:
    AsyncOpenAI = None  # type: ignore
    APIConnectionError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore
    InternalServerError = Exception  # type: ignore
    APIStatusError = Exception  # type: ignore

from llm.cache import DiskCache

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ModelConfig(BaseModel):
    id: str
    role: str
    context_window: int
    max_output_tokens: int
    temperature: float = 0.2
    cost_per_1m_input_tokens_usd: float = 0.0
    cost_per_1m_output_tokens_usd: float = 0.0


def load_dotenv_if_present(env_path: str | Path = ".env") -> None:
    """Load environment variables from a local .env file if present, without external dependencies."""
    p = Path(env_path)
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k not in os.environ:
                os.environ[k] = v


# Automatically attempt to load local .env if present
load_dotenv_if_present()


class ModelConfigManager:
    """
    Loads and manages model IDs strictly from config/models.yaml (Rule 3).
    Supports provider profiles for Nebius Token Factory, Groq, and OpenRouter.
    """

    def __init__(self, config_path: str | Path = "config/models.yaml", provider: Optional[str] = None):
        load_dotenv_if_present()
        self.config_path = Path(config_path)
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Model configuration file not found at {self.config_path}. "
                "Per Rule 3, model IDs must be read from config/models.yaml."
            )
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.models: Dict[str, ModelConfig] = {}
        for role, spec in data.get("models", {}).items():
            self.models[role] = ModelConfig(**spec)

        self.providers = data.get("providers", {})
        self.endpoints = data.get("endpoints", {})

        # Active provider determination
        active_provider = (
            provider
            or os.environ.get("LLM_PROVIDER")
            or ("groq" if os.environ.get("GROQ_API_KEY") else None)
            or ("openrouter" if os.environ.get("OPENROUTER_API_KEY") else None)
        )
        if active_provider and active_provider.lower() in self.providers and not os.environ.get("NEBIUS_API_KEY"):
            p_info = self.providers[active_provider.lower()]
            p_models = p_info.get("models", {})
            for role, mid in p_models.items():
                if role in self.models:
                    self.models[role] = self.models[role].model_copy(update={"id": mid})
            if "base_url" in p_info:
                self.endpoints["base_url"] = p_info["base_url"]

    def get_model(self, role_or_id: str) -> ModelConfig:
        if role_or_id in self.models:
            return self.models[role_or_id]
        for cfg in self.models.values():
            if cfg.id == role_or_id:
                return cfg
        raise KeyError(
            f"Unknown model or role '{role_or_id}'. "
            f"Allowed roles: {list(self.models.keys())}. "
            "Never invent model IDs (Rule 3)."
        )


def repair_truncated_json(raw_text: str) -> Optional[str]:
    """
    Attempt heuristic recovery of JSON truncated by token limits.
    Handles unclosed strings, unmatched brackets '[' and braces '{',
    and trailing commas or colons.
    """
    text = raw_text.strip()
    if not text:
        return None

    # Strip markdown code blocks if present
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    stack: List[str] = []
    in_string = False
    escape = False

    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch in "{[":
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

    repaired = text
    if in_string:
        repaired += '"'

    repaired_strip = repaired.rstrip()
    if repaired_strip.endswith(":"):
        repaired_strip += " null"
    elif repaired_strip.endswith(","):
        repaired_strip = repaired_strip[:-1].rstrip()
    repaired = repaired_strip

    # Close remaining unmatched openers in reverse order
    for opener in reversed(stack):
        if opener == "{":
            repaired += "}"
        elif opener == "[":
            repaired += "]"

    try:
        json.loads(repaired)
        return repaired
    except Exception:
        return None


class LLMResponse(BaseModel):
    content: str
    reasoning: Optional[str] = None
    model: str
    role: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cached: bool = False
    cost_estimate_usd: float = 0.0


class LLMClient:
    """
    OpenAI-compatible client configured for Nebius Token Factory with Nemotron models.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache: Optional[DiskCache] = None,
        concurrency_limit: int = 10,
        max_retries: int = 3,
        timeout_seconds: float = 60.0,
        mock_provider: Optional[Any] = None,
        config_path: str | Path = "config/models.yaml",
    ):
        self.config_manager = ModelConfigManager(config_path=config_path)
        self.mock_provider = mock_provider
        self.cache = cache or DiskCache()
        self.concurrency_limit = concurrency_limit
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.semaphore = asyncio.Semaphore(concurrency_limit)

        self.call_history: List[Dict[str, Any]] = []
        self.last_call_log: Dict[str, Any] = {}

        if mock_provider is None:
            # Rule 7: Read secrets strictly from environment
            resolved_key = (
                api_key
                or os.environ.get("NEBIUS_API_KEY")
                or os.environ.get("GROQ_API_KEY")
                or os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            resolved_url = (
                base_url
                or os.environ.get("NEBIUS_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL")
                or self.config_manager.endpoints.get("base_url", "https://api.tokenfactory.nebius.com/v1")
            )
            if not resolved_key:
                # If no key is set and no mock provider, we record it and will raise upon live call
                self._client = None
            else:
                if AsyncOpenAI is None:
                    raise ImportError("The openai package is required for live LLM calls. Install via pip install openai.")
                self._client = AsyncOpenAI(
                    base_url=resolved_url,
                    api_key=resolved_key,
                    timeout=timeout_seconds,
                )
        else:
            self._client = None

    def calculate_cost(self, model_cfg: ModelConfig, prompt_tokens: int, completion_tokens: int) -> float:
        input_cost = (prompt_tokens / 1_000_000.0) * model_cfg.cost_per_1m_input_tokens_usd
        output_cost = (completion_tokens / 1_000_000.0) * model_cfg.cost_per_1m_output_tokens_usd
        return round(input_cost + output_cost, 6)

    def _extract_reasoning_and_content(
        self,
        message: Any,
        finish_reason: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        """
        Extract content and reasoning across diverse reasoning-model output shapes:
        - standard content field
        - reasoning / reasoning_content fields
        - inline <thought> tags
        - answer only in reasoning_content/reasoning when content is empty
        - empty content with finish_reason='length'
        """
        content = ""
        reasoning = None

        if hasattr(message, "content") and message.content:
            content = str(message.content).strip()
        elif isinstance(message, dict) and "content" in message and message["content"] is not None:
            content = str(message["content"]).strip()

        # Check for reasoning field from SDK / API / dictionary
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            reasoning = str(message.reasoning_content).strip()
        elif hasattr(message, "reasoning") and message.reasoning:
            reasoning = str(message.reasoning).strip()
        elif isinstance(message, dict):
            if "reasoning_content" in message and message["reasoning_content"]:
                reasoning = str(message["reasoning_content"]).strip()
            elif "reasoning" in message and message["reasoning"]:
                reasoning = str(message["reasoning"]).strip()

        # Check for inline <thought> tags if reasoning was not returned separately
        if not reasoning and "<thought>" in content and "</thought>" in content:
            start = content.find("<thought>") + len("<thought>")
            end = content.find("</thought>")
            reasoning = content[start:end].strip()
            content = (content[:start - len("<thought>")] + content[end + len("</thought>"):]).strip()

        # Handle empty content with finish_reason='length'
        if finish_reason == "length" and not content:
            err_msg = (
                "LLM generation truncated: output reached max_tokens (finish_reason='length') "
                "before emitting content."
            )
            if reasoning:
                err_msg += f" (Partial reasoning: {reasoning[:300]})"
            raise RuntimeError(err_msg)

        # Handle answer only in reasoning_content or reasoning when content is empty
        if not content and reasoning:
            # Check if there is an embedded JSON or code block in reasoning
            if "```json" in reasoning:
                s_idx = reasoning.find("```json") + len("```json")
                e_idx = reasoning.find("```", s_idx)
                content = reasoning[s_idx:e_idx].strip() if e_idx != -1 else reasoning[s_idx:].strip()
            elif "```" in reasoning:
                s_idx = reasoning.find("```") + len("```")
                e_idx = reasoning.find("```", s_idx)
                content = reasoning[s_idx:e_idx].strip() if e_idx != -1 else reasoning[s_idx:].strip()
            else:
                content = reasoning

        return content, reasoning

    async def generate(
        self,
        role: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        **extra_params: Any,
    ) -> LLMResponse:
        """
        Execute an async LLM generation with concurrency limiting, retries,
        disk caching, and metric logging.
        """
        model_cfg = self.config_manager.get_model(role)
        temp = temperature if temperature is not None else model_cfg.temperature
        tokens_limit = max_tokens if max_tokens is not None else model_cfg.max_output_tokens

        params = {
            "temperature": temp,
            "max_tokens": tokens_limit,
            **extra_params,
        }

        # 1. Check disk cache
        cache_key = self.cache.compute_key(model_cfg.id, messages, params)
        if use_cache and self.cache.has(cache_key):
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                cost = self.calculate_cost(
                    model_cfg,
                    cached_data.get("prompt_tokens", 0),
                    cached_data.get("completion_tokens", 0),
                )
                response = LLMResponse(
                    content=cached_data.get("content", ""),
                    reasoning=cached_data.get("reasoning"),
                    model=model_cfg.id,
                    role=role,
                    prompt_tokens=cached_data.get("prompt_tokens", 0),
                    completion_tokens=cached_data.get("completion_tokens", 0),
                    latency_ms=0.0,
                    cached=True,
                    cost_estimate_usd=cost,
                )
                self._record_log(response, cached=True)
                return response

        # 2. Acquire semaphore for concurrency limiting
        async with self.semaphore:
            start_time = time.perf_counter()
            attempt = 0
            backoff_delay = 0.5

            while attempt <= self.max_retries:
                try:
                    if self.mock_provider is not None:
                        # Invoke mock provider
                        raw_result = await self.mock_provider.generate(
                            model_id=model_cfg.id,
                            role=role,
                            messages=messages,
                            **params,
                        )
                        finish_reason = raw_result.get("finish_reason") if isinstance(raw_result, dict) else None
                        content, reasoning = self._extract_reasoning_and_content(raw_result, finish_reason=finish_reason)
                        prompt_tokens = raw_result.get("prompt_tokens", 0) if isinstance(raw_result, dict) else 0
                        completion_tokens = raw_result.get("completion_tokens", 0) if isinstance(raw_result, dict) else 0
                    else:
                        if self._client is None:
                            raise ValueError(
                                "NEBIUS_API_KEY not configured in environment or constructor. "
                                "Per Rule 7, secrets must be read from environment."
                            )
                        completion = await self._client.chat.completions.create(
                            model=model_cfg.id,
                            messages=messages,  # type: ignore
                            temperature=temp,
                            max_tokens=tokens_limit,
                            **extra_params,
                        )
                        choice = completion.choices[0]
                        finish_reason = getattr(choice, "finish_reason", None)
                        content, reasoning = self._extract_reasoning_and_content(
                            choice.message, finish_reason=finish_reason
                        )
                        prompt_tokens = completion.usage.prompt_tokens if completion.usage else 0
                        completion_tokens = completion.usage.completion_tokens if completion.usage else 0

                    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                    cost = self.calculate_cost(model_cfg, prompt_tokens, completion_tokens)

                    # Store in cache
                    if use_cache:
                        self.cache.set(
                            cache_key,
                            {
                                "content": content,
                                "reasoning": reasoning,
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "model": model_cfg.id,
                            },
                        )

                    response = LLMResponse(
                        content=content,
                        reasoning=reasoning,
                        model=model_cfg.id,
                        role=role,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        cached=False,
                        cost_estimate_usd=cost,
                    )
                    self._record_log(response, cached=False)
                    return response

                except Exception as e:
                    attempt += 1
                    is_retryable = (
                        isinstance(e, (APIConnectionError, RateLimitError, InternalServerError))
                        or "rate limit" in str(e).lower()
                        or "429" in str(e)
                        or "503" in str(e)
                    )
                    if attempt > self.max_retries or not is_retryable:
                        logger.error(f"LLM call failed after {attempt} attempts: {e}")
                        # Rule 5: No bare except and no silent fallbacks. Fail loudly with a clear error.
                        raise RuntimeError(f"LLM generation failed for {role} ({model_cfg.id}): {e}") from e

                    # Extract Retry-After if present (e.g. from HTTP 429 response headers)
                    retry_after_sec: Optional[float] = None
                    resp_obj = getattr(e, "response", None)
                    headers = getattr(resp_obj, "headers", None) or getattr(e, "headers", None)
                    if headers and hasattr(headers, "get"):
                        val = headers.get("retry-after") or headers.get("Retry-After")
                        if val is not None:
                            try:
                                retry_after_sec = float(val)
                            except (ValueError, TypeError):
                                pass
                    elif hasattr(e, "retry_after"):
                        try:
                            retry_after_sec = float(getattr(e, "retry_after"))
                        except (ValueError, TypeError):
                            pass

                    sleep_duration = max(retry_after_sec or 0.0, backoff_delay)
                    if retry_after_sec is not None:
                        logger.warning(
                            f"HTTP 429 rate limit received with Retry-After={retry_after_sec}s. "
                            f"Sleeping for {sleep_duration}s."
                        )

                    await asyncio.sleep(sleep_duration)
                    backoff_delay *= 2.0

            raise RuntimeError(f"Exceeded max retries ({self.max_retries}) for model {role}")

    def _record_log(self, response: LLMResponse, cached: bool) -> None:
        log_entry = {
            "model": response.model,
            "role": response.role,
            "tokens": {
                "prompt": response.prompt_tokens,
                "completion": response.completion_tokens,
                "total": response.prompt_tokens + response.completion_tokens,
            },
            "latency_ms": response.latency_ms,
            "cached": cached,
            "cost_usd": response.cost_estimate_usd,
            "has_reasoning": response.reasoning is not None and len(response.reasoning) > 0,
        }
        self.last_call_log = log_entry
        self.call_history.append(log_entry)
        logger.info(
            f"[LLM] role={response.role} model={response.model} cached={cached} "
            f"tokens={response.prompt_tokens}+{response.completion_tokens} "
            f"latency={response.latency_ms}ms cost=${response.cost_estimate_usd:.6f}"
        )

    async def generate_structured(
        self,
        role: str,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        max_repairs: int = 2,
        **kwargs: Any,
    ) -> T:
        """
        Validate LLM output against a Pydantic schema with an automatic repair loop (Rule 5 & 6).
        """
        history = list(messages)
        # Append instruction to return pure JSON
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        history.append({
            "role": "system",
            "content": f"You must respond ONLY with a valid JSON object strictly matching this schema:\n{schema_json}",
        })

        for repair_attempt in range(max_repairs + 1):
            response = await self.generate(role=role, messages=history, **kwargs)
            raw_text = response.content.strip()

            # Strip markdown json blocks if present
            if raw_text.startswith("```json"):
                raw_text = raw_text[len("```json"):].strip()
            if raw_text.startswith("```"):
                raw_text = raw_text[len("```"):].strip()
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()

            try:
                data = json.loads(raw_text)
                validated = response_model.model_validate(data)
                return validated
            except (json.JSONDecodeError, ValidationError) as err:
                # If direct parse failed, attempt heuristic repair for truncated JSON
                if isinstance(err, json.JSONDecodeError):
                    repaired_json = repair_truncated_json(raw_text)
                    if repaired_json:
                        try:
                            data = json.loads(repaired_json)
                            validated = response_model.model_validate(data)
                            logger.info("Successfully recovered truncated JSON via heuristic bracket repair.")
                            return validated
                        except Exception:
                            pass

                if repair_attempt >= max_repairs:
                    # Rule 5: Fail loudly with a clear error
                    raise ValidationError.from_exception_data(
                        title=response_model.__name__,
                        line_errors=[
                            {
                                "type": "value_error",
                                "loc": ("root",),
                                "input": raw_text,
                                "ctx": {"error": f"Failed validation after {max_repairs} repairs: {err}"},
                            }
                        ],
                    )

                logger.warning(
                    f"Structured validation failed on attempt {repair_attempt + 1}. Requesting repair. Error: {err}"
                )
                # Append repair prompt
                history.append({"role": "assistant", "content": response.content})
                is_truncated = isinstance(err, json.JSONDecodeError) and (
                    raw_text.endswith(",") or raw_text.endswith(":") or raw_text.count("{") > raw_text.count("}")
                )
                repair_msg = (
                    f"Your output failed validation against the schema with the following error:\n{err}\n"
                )
                if is_truncated:
                    repair_msg += (
                        "Notice: Your previous response was truncated mid-JSON. "
                        "Please provide the complete, closed JSON object strictly matching the schema."
                    )
                else:
                    repair_msg += "Please correct the JSON and return ONLY the valid JSON object without commentary."

                history.append({
                    "role": "user",
                    "content": repair_msg,
                })

        raise RuntimeError("Unexpected exit from repair loop.")
