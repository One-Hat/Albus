# Platform & Developer Experience Feedback: Nebius Token Factory & Serverless AI

### 1. Mandatory Funded Balance Before API Key Generation
- **Feedback**: Requiring a pre-funded credit card / deposit balance before allowing users to generate an API key significantly slowed down hackathon development.
- **Impact**: In rapid 48-to-72-hour hackathons (and developer evaluation cycles), teams often finalize code and tests while awaiting corporate card or reimbursement approvals. Without the ability to generate a rate-limited trial key or sandbox token, live integration testing was blocked for 24–48 hours, forcing the entire pipeline to rely on offline synthetic fixtures and mock emulation.
- **Recommendation**: Provide a complimentary, zero-deposit sandbox quota (e.g., $5.00 in trial credits or a strictly rate-limited 1 req/s test key) immediately upon developer registration so developers can verify API compatibility within minutes.

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
