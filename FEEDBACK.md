# Platform & Developer Experience Feedback: Nebius Token Factory & Serverless AI

### 1. Mandatory Funded Balance & Credit Card Gating for API Keys
- **Friction Encountered**: Requiring a credit card registration and pre-funded deposit balance before issuing an API key blocked rapid development and live verification during the hackathon.
- **Developer Impact**: In high-velocity 48–72 hour hackathon timelines, developers need immediate API connectivity to test token throughput, context chunking, and reasoning response formats. When team members lack a corporate credit card or wait for billing authorizations, all live calls stall, forcing the team to build offline mock harnesses and synthetic fixtures.
- **ALBUS Solution & Open-Source Fallback**: Rather than stopping, we engineered ALBUS with a dual-provider architecture in `src/llm/client.py`. We integrated **Groq Cloud** as a zero-friction open-source model provider (using `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, and Llama-3 open weights) via `GROQ_API_KEY`. This allowed us to execute live reasoning smoke tests, test latency, and run benchmarks end-to-end without credit card friction, while keeping the production Nebius Serverless Endpoints and NVIDIA Nemotron configurations 100% ready for deployment once funded.
- **Recommendation for Nebius**: Provide an instant, zero-deposit developer sandbox quota (e.g. $5.00 complimentary credits or a rate-limited 1 req/s evaluation tier) upon GitHub/Google authentication so developers can prototype immediately.

---

### 2. Reasoning-Model Response Shape Standardization
- **Feedback**: Nemotron and open-weights reasoning models exhibit variable output shapes depending on inference engine settings:
  1. Standard `message.content`
  2. Distinct `message.reasoning_content` (with empty `content`)
  3. Inline `<thought>` tags within `content`
- **Recommendation**: Standardize OpenAI-compatible API responses with explicit documentation and guarantees that structured JSON outputs (e.g., via `response_format={"type": "json_object"}`) will place the final JSON in `message.content` while streaming internal chain-of-thought in `reasoning_content`.

---

### 3. Serverless Jobs Documentation on CPU-Only Workloads
- **Feedback**: The Nebius Serverless Jobs CLI documentation extensively details GPU platforms (e.g., `--platform gpu-l40s-a`, `--preset 1gpu-8vcpu-32gb`), but CPU-only platforms for lightweight orchestration, backtest evaluation, and parser workers are undocumented (`UNKNOWN`).
- **Recommendation**: Document CPU-only presets and platforms directly in the Serverless Jobs quickstart guide for cost-effective batch data pipelines that do not require GPU hardware.

---

### 2. Reasoning-Model Response Shape Standardization
- **Feedback**: Nemotron reasoning models exhibit variable output shapes depending on inference engine settings:
  1. Standard `message.content`
  2. Distinct `message.reasoning_content` (with empty `content`)
  3. Inline `<thought>` tags within `content`
- **Recommendation**: Standardize OpenAI-compatible API responses with explicit documentation and guarantees that structured JSON outputs (e.g., via `response_format={"type": "json_object"}`) will place the final JSON in `message.content` while streaming internal chain-of-thought in `reasoning_content`.

---

### 3. Serverless Jobs Documentation on CPU-Only Workloads
- **Feedback**: The Nebius Serverless Jobs CLI documentation extensively details GPU platforms (e.g., `--platform gpu-l40s-a`, `--preset 1gpu-8vcpu-32gb`), but CPU-only platforms for lightweight orchestration, backtest evaluation, and parser workers are undocumented (`UNKNOWN`).
- **Recommendation**: Document CPU-only presets and platforms directly in the Serverless Jobs quickstart guide for cost-effective batch data pipelines that do not require GPU hardware.
