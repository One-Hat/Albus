# Vercel Deployment Feasibility Assessment: Albus Backend

This document evaluates the technical feasibility of deploying the **Albus Quantitative Research & Forecasting Swarm Backend** as Python Serverless Functions on Vercel.

---

## 1. Specification & Metric Limits Matrix

| Parameter | Vercel Serverless (Python) Specification | Albus Pipeline Requirement | Compatibility Status | Official Vercel Documentation Citation |
|---|---|---|---|---|
| **Runtime Version** | Python 3.12, 3.9 | Requires `>=3.11, <3.15` (`pyproject.toml`) | **COMPATIBLE** | [Vercel Python Runtime Docs](https://vercel.com/docs/functions/runtimes/python) |
| **Max Execution Duration (Hobby)** | 300 seconds (5 minutes) | Replay API: < 100ms; Live SEC Swarm: 60–180s | **FEASIBLE** | [Vercel Function Duration Limits](https://vercel.com/docs/functions/configuring-functions/duration) |
| **Max Execution Duration (Pro/Enterprise)** | Default 300s, configurable up to 800s (standard) or 1,800s (beta Fluid Compute) | Full 20-ticker backtest + 3-round Swarm: ~120–240s | **COMPATIBLE** | [Vercel Fluid Compute Docs](https://vercel.com/docs/functions/fluid-compute) |
| **Standard Bundle Size Limit** | 500 MB uncompressed (read-only filesystem) | Core dependencies (`numpy`, `pandas`, `scipy`, `scikit-learn`, `fastapi`, `pydantic`): ~260 MB | **COMPATIBLE** | [Vercel Function Bundle Limits](https://vercel.com/docs/functions/serverless-functions/runtimes#bundle-size) |
| **Large Functions Extension** | Up to 5 GB uncompressed (`VERCEL_SUPPORT_LARGE_FUNCTIONS=1`) | Not strictly needed, but provides buffer for ML wheels | **COMPATIBLE** | [Vercel Large Functions Guide](https://vercel.com/docs/functions/fluid-compute#large-functions) |
| **Scratch Space (`/tmp`)** | 500 MB writable ephemeral storage | SEC EDGAR raw 10-Ks: 54 filings * ~1.5 MB = ~81 MB | **FEASIBLE** (ephemeral only) | [Vercel Functions Filesystem](https://vercel.com/docs/functions/runtimes#filesystem) |
| **Response Streaming (SSE)** | Supported natively via ASGI chunked transfer / FastAPI `StreamingResponse` | `GET /api/run/{id}/events` SSE pipeline events | **COMPATIBLE** | [Vercel Streaming Responses](https://vercel.com/docs/functions/streaming) |
| **Environment Variable Limits** | Combined total size max 64 KB; up to 1,000 variables | Current `.env`: ~300 bytes (`GROQ_API_KEY`, `SEC_USER_AGENT`, `LIVE_ENABLED`) | **COMPATIBLE** | [Vercel Environment Variables Limits](https://vercel.com/docs/projects/environment-variables#limits) |
| **Background Threading / Daemon** | Terminates upon request completion unless streaming | Swarm background workers without open HTTP stream: **NOT SUPPORTED** | **CONSTRAINT** | [Vercel Serverless Lifecycle](https://vercel.com/docs/functions/serverless-functions/lifecycle) |
| **Persistent Local Disk Cache** | Ephemeral only (`/tmp` purged on cold shutdown) | Disk cache (`.cache/llm_http/`) resets across cold boots | **CONSTRAINT** | [Vercel Ephemeral Storage Docs](https://vercel.com/docs/functions/runtimes#filesystem) |
| **Concurrency Scaling & Burst Limits** | Plan dependent; exact cold-start concurrency scale rate for Python: **UNKNOWN** | Single concurrent run per client IP enforced at app level | **NEEDS MONITORING** | *UNKNOWN in public docs* |

---

## 2. Dependency Sizing Analysis

Our current production dependencies (`pyproject.toml`):
- `pydantic` (~15 MB)
- `fastapi` + `starlette` + `uvicorn` (~20 MB)
- `numpy` (~35 MB)
- `pandas` (~65 MB)
- `scipy` (~85 MB)
- `scikit-learn` (~45 MB)
- `yfinance` + `httpx` + `beautifulsoup4` + `lxml` (~25 MB)

**Total Uncompressed Disk Footprint:** **~290 MB**
- Under standard 500 MB ceiling (58% of quota).
- Fits cleanly within Vercel's standard Python serverless bundle limit.
- Cold start overhead: ~3.5 to 6.5 seconds for initial Python wheel loading into memory.

---

## 3. Streaming (SSE) Feasibility

Vercel Serverless Functions running on the Python ASGI runtime support HTTP streaming via chunked transfer encoding:
- `FastAPI`'s `StreamingResponse(event_generator(), media_type="text/event-stream")` works out-of-the-box.
- Header requirements: `Cache-Control: no-cache` and `Connection: keep-alive`.
- **Limitation:** In serverless execution, the streaming connection must remain continuously open. If the client disconnects or network drops, Vercel kills the container immediately, terminating any active in-flight swarm calculation.

---

## 4. Configuration Template (`vercel.json`)

If deployed on Vercel, the root `vercel.json` must configure:

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "functions": {
    "api/index.py": {
      "runtime": "python3.12",
      "maxDuration": 300,
      "memory": 1024
    }
  },
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "api/index.py"
    }
  ]
}
```

---

## 5. Architectural Recommendation

### Option A: Hybrid Architecture (**RECOMMENDED**)
1. **Frontend (`web/`):** Deployed on **Vercel** (Next.js/React App Router on Vercel Global Edge Network).
2. **Replay Mode & Static Metadata (`web/public/replays/`):** Served directly via Vercel Edge CDN with zero cold starts, instantaneous sub-50ms latency, and 100% contract compliance.
3. **Lightweight Read APIs (`/api/config`, `/api/runs`, `/api/runs/{id}`, `/api/ledger/verify/{id}`):** Can run directly on Vercel Serverless Function (`api/index.py`).
4. **Heavy Live Execution Swarm (`POST /api/run`):** Hosted on a dedicated container runner (e.g. **Nebius Serverless Job**, Docker on Render/Modal/Fly.io) because:
   - Full 20-ticker SEC download and multi-round LLM swarm reasoning can exceed ephemeral `/tmp` limits and 300s Hobby timeout.
   - Long-running disk caching (`.cache/llm_http`) is preserved across invocations, dramatically reducing Groq API token burn.

### Option B: Monolithic All-on-Vercel Architecture
- Feasible **IF AND ONLY IF** live swarm runs are capped at `<= 5 tickers` and `<= 200s` wall time on a Pro plan with Fluid Compute enabled.
- High risk of cold-start lag (~5s) and `/tmp` cache evaporation between runs.

---

*Awaiting collaborator / repository owner decision on Option A vs Option B before applying final Vercel deployment manifests.*
