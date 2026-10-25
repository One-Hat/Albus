# AGENTS.md - Quant Research Copilot + Self-Calibrating Forecasting Swarm

1. Work on one phase at a time. Touch only files the phase names.
2. Write tests first. Run them and paste the real output into the walkthrough.
3. Never invent model IDs. Read them from config/models.yaml.
4. MockLLM may only be imported under tests/. Never in src/.
5. No bare except and no silent fallbacks. Fail loudly with a clear error.
6. Every LLM output is validated by pydantic. Quotes must be verbatim substrings.
7. No secrets in code or logs. Read them from the environment.
8. Benchmark numbers come only from scripts/run_benchmarks.py output.
9. Keep test_no_lookahead green at all times.
10. If a requirement is ambiguous, stop and ask. Do not guess.
