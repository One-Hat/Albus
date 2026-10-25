# ALBUS: Quant Research Copilot + Self-Calibrating Forecasting Swarm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Track:** Best Apps and Agents  
**Nebius x NVIDIA Global AI Hackathon**

## Overview
ALBUS is a quant research copilot paired with a self-calibrating forecasting swarm. You type a research hypothesis in plain English; the system extracts signals from SEC filings (Item 1A risk factors) and market data, tests the idea on historical price series, and turns the signal into live, falsifiable forecasts scored over time in an append-only, hash-chained ledger.

### Architecture Highlights
- **Nemotron 3 Nano (`nvidia/nemotron-3-nano-30b-a3b`):** High-throughput bulk extraction of filing diffs with verbatim quote verification and 8 diverse forecasting swarm personas.
- **Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`):** Research agent reasoning, backtest execution tool loop, and swarm aggregation.
- **Nemotron 3 Ultra (`nvidia/Nemotron-3-Ultra-550b-a55b`):** Skeptical critic and final adjudicator running deterministic lookahead/cost checks and adversarial audit.
- **Nebius Token Factory:** High-speed inference with sub-second latency and persistent disk caching.

## Project Structure
```
/
  config/models.yaml      # Model configurations (Nano, Super, Ultra)
  src/
    llm/                  # Token Factory async client, disk cache, mock provider
    data/                 # EDGAR SEC filing ingestion & yfinance price loader
    extract/              # Item 1A diff extractor & verbatim quote check
    backtest/             # Walk-forward backtest engine with no-lookahead verification
    research/             # Super research agent & tool loop
    swarm/                # Forecasting personas, hash-chained ledger, calibrator
    critic/               # Ultra critic with deterministic checks
    report/               # Report generator
    app/                  # Interactive Streamlit dashboard
  tests/                  # Strict unit, integration, and planted-flaw test suites
  scripts/                # Model discovery, benchmarks, and secret scanning
```
