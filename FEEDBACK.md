# Developer Feedback: Nebius Token Factory & NVIDIA Nemotron Models

This document logs practical observations, architectural considerations, and developer experience friction points encountered while developing **ALBUS** on the **Nebius Token Factory** and **NVIDIA Nemotron** ecosystem.

---

## 1. NVIDIA Nemotron Model Roles & Reasoning Tokens

### Observation: Dedicated Reasoning Field vs. Content
- **Friction Point:** Frontier reasoning models (such as Nemotron Super and Ultra) emit internal reasoning tokens before or alongside their target output. Standard OpenAI-compatible clients that inspect only `message.content` frequently receive empty payloads or strings interspersed with chain-of-thought tokens.
- **Solution in ALBUS:** We implemented custom extraction logic in [`src/llm/client.py`](file:///d:/NVIDIA/src/llm/client.py) that safely bifurcates `reasoning` / `reasoning_content` from `content`, as well as parsing inline `<thought>...</thought>` blocks. This guarantees downstream Pydantic parsers receive pure JSON without corrupting quantitative filing diffs.
- **Feedback for Nebius / NVIDIA:** Exposing explicit `reasoning` attributes in API responses is exceptionally valuable for auditing agent steps without having to regex-strip system thoughts.

---

## 2. Multi-Tier Model Hierarchy (Nano vs. Super vs. Ultra)

### Observation: Latency, Cost, and Specialization
- **Nemotron 3 Nano (`nvidia/nemotron-3-nano-30b-a3b`):** 
  - Ideal for high-throughput, parallel extraction tasks (such as parsing Item 1A risk factors across dozens of 10-K filings simultaneously) and running high-diversity forecasting personas.
  - Sub-second generation speed makes high fan-out swarm workflows economically feasible.
- **Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`):**
  - Robust middle-tier reasoner suited for structured tool calling, quantitative hypothesis generation, and aggregation.
- **Nemotron 3 Ultra (`nvidia/Nemotron-3-Ultra-550b-a55b`):**
  - Frontier reasoning capabilities reserved for the final adversarial audit and skeptical critic review.

---

## 3. Rate Limits, Concurrency, and Caching

### Observation: High-Volume Multi-Agent Fan-Out
- Running a 10-persona forecasting swarm across multiple annual filings generates spikes of 100+ API calls in rapid succession.
- **Solution in ALBUS:** We added an asynchronous concurrency semaphore (`asyncio.Semaphore(10)`) and an exponential backoff retry loop (`max_retries=3`) to gracefully absorb burst rate limits.
- **Deterministic Disk Cache:** To avoid burning tokens during local development and regression testing, a SHA-256 disk cache (`.cache/llm/`) stores responses keyed by `(model, prompt, params)`.

---

## 4. SEC EDGAR Ingestion & Data Hygiene

### Observation: Strict Timestamping & Fair-Access Compliance
- To prevent lookahead bias in algorithmic backtests, filings must be tagged with their exact **SEC Acceptance Timestamp** (Eastern Time) rather than file dates.
- All outbound requests to SEC EDGAR enforce declared `User-Agent: Sample Company Name Contact@domain.com` headers and a rate limiter capped at $\le 10$ requests per second.
