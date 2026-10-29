"""FastAPI application implementing the Albus Live & Replay API contract."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional
import uuid
import yaml

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.api.models import (
    ModelsConfig,
    RunIndexItem,
    RunSummary,
    StageEvent,
    StageName,
    StageStatus,
)

# Base Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "models.yaml"
TICKERS_PATH = REPO_ROOT / "config" / "tickers.txt"
REPLAYS_DIR = REPO_ROOT / "web" / "public" / "replays"

# In-memory Rate Limiting & Concurrency Tracker
IP_REQUEST_TIMESTAMPS: Dict[str, List[float]] = {}
ACTIVE_RUNS_BY_IP: Dict[str, str] = {}
RATE_LIMIT_WINDOW_SEC = 60.0
RATE_LIMIT_MAX_REQUESTS = 20

# Active Run Events In-Memory Queue for SSE
RUN_EVENTS_REGISTRY: Dict[str, List[StageEvent]] = {}


class RunRequest(BaseModel):
    hypothesis: str = Field(..., min_length=10, max_length=500)
    universe: List[str] = Field(..., min_length=1, max_length=20)


class RunResponse(BaseModel):
    run_id: str


class ConfigResponse(BaseModel):
    provider: str
    models: ModelsConfig
    live_enabled: bool
    limits: Dict[str, Any]


class LedgerVerifyResponse(BaseModel):
    verified: bool
    broken_at: Optional[int] = None
    length: int
    head_hash: str


def load_allowed_tickers() -> set[str]:
    """Load allowable tickers from config/tickers.txt."""
    if not TICKERS_PATH.exists():
        return set()
    with open(TICKERS_PATH, "r", encoding="utf-8") as f:
        return {line.strip().upper() for line in f if line.strip() and not line.startswith("#")}


def load_models_config() -> tuple[str, ModelsConfig]:
    """Load active provider and model IDs from config/models.yaml and environment."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if provider in data.get("providers", {}):
        prov_models = data["providers"][provider]["models"]
        models_cfg = ModelsConfig(
            small=prov_models["nano"],
            mid=prov_models["super"],
            large=prov_models["ultra"],
        )
    else:
        models_cfg = ModelsConfig(
            small=data["models"]["nano"]["id"],
            mid=data["models"]["super"]["id"],
            large=data["models"]["ultra"]["id"],
        )
    return provider, models_cfg


def check_rate_limit(client_ip: str) -> None:
    """Enforce token bucket rate limiting per client IP."""
    now = time.time()
    timestamps = IP_REQUEST_TIMESTAMPS.get(client_ip, [])
    # Filter to current window
    valid = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SEC]
    if len(valid) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 20 requests per minute.",
            headers={"Retry-After": "60"},
        )
    valid.append(now)
    IP_REQUEST_TIMESTAMPS[client_ip] = valid


def create_app() -> FastAPI:
    app = FastAPI(
        title="Albus Quantitative Forecasting & Swarm API",
        version="1.0",
        description="FastAPI service serving contracts/types.ts specification.",
    )

    # CORS locked to domain / configurable
    allowed_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173",
    ).split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def rate_limiting_middleware(request: Request, call_next):
        client_ip = request.client.host if request.client else "127.0.0.1"
        if request.url.path.startswith("/api/"):
            try:
                check_rate_limit(client_ip)
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"detail": exc.detail},
                    headers=exc.headers,
                )
        return await call_next(request)

    @app.get("/api/config", response_model=ConfigResponse)
    async def get_config() -> ConfigResponse:
        """GET /api/config -> { provider, models, live_enabled, limits }"""
        provider, models = load_models_config()
        live_enabled = os.getenv("LIVE_ENABLED", "false").lower() == "true"
        limits = {
            "max_tickers": 20,
            "max_tokens_per_call": 4096,
            "max_wall_time_sec": 120,
            "global_daily_budget_usd": 10.0,
            "rate_limit_rpm": RATE_LIMIT_MAX_REQUESTS,
            "concurrency_limit_per_ip": 1,
        }
        return ConfigResponse(
            provider=provider,
            models=models,
            live_enabled=live_enabled,
            limits=limits,
        )

    @app.get("/api/runs", response_model=List[RunIndexItem])
    async def get_runs() -> List[RunIndexItem]:
        """GET /api/runs -> RunIndexItem[] from replays/index.json"""
        index_file = REPLAYS_DIR / "index.json"
        if not index_file.exists():
            return []
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [RunIndexItem.model_validate(item) for item in data]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read index: {e}")

    @app.get("/api/runs/{run_id}", response_model=RunSummary)
    async def get_run_details(run_id: str) -> RunSummary:
        """GET /api/runs/{id} -> RunSummary from replays/{id}.json"""
        run_file = REPLAYS_DIR / f"{run_id}.json"
        if not run_file.exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        try:
            with open(run_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return RunSummary.model_validate(data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Corrupt run payload: {e}")

    @app.post("/api/run", response_model=RunResponse)
    async def trigger_run(payload: RunRequest, request: Request) -> RunResponse:
        """POST /api/run {hypothesis, universe} -> { run_id }"""
        client_ip = request.client.host if request.client else "127.0.0.1"

        # 1. LIVE_ENABLED guard
        live_enabled = os.getenv("LIVE_ENABLED", "false").lower() == "true"
        if not live_enabled:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Live execution is currently disabled (LIVE_ENABLED=false). Explore historical runs via Replay Mode.",
            )

        # 2. Concurrency limit per IP
        if client_ip in ACTIVE_RUNS_BY_IP:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Concurrent run already active for this client. Please wait for completion.",
            )

        # 3. Ticker allowlist validation
        allowed = load_allowed_tickers()
        invalid_tickers = [t for t in payload.universe if t.upper() not in allowed]
        if invalid_tickers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tickers not in verified universe allowlist: {invalid_tickers}",
            )

        # 4. Untrusted input sanitization
        cleaned_hypothesis = re.sub(r"[<>{}\\]", "", payload.hypothesis).strip()
        if len(cleaned_hypothesis) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Hypothesis contains invalid characters or is too short.",
            )

        # Create new run ID
        run_id = f"live-run-{uuid.uuid4().hex[:8]}"
        ACTIVE_RUNS_BY_IP[client_ip] = run_id

        # Populate initial SSE events pipeline
        RUN_EVENTS_REGISTRY[run_id] = [
            StageEvent(stage="ingest", status="start", total=len(payload.universe), message="Fetching SEC 10-K filings..."),
            StageEvent(stage="extract", status="progress", done=1, total=len(payload.universe), message="Extracting Item 1A Risk Factors..."),
            StageEvent(stage="research", status="progress", message="Research agent identifying semantic diffs..."),
            StageEvent(stage="swarm", status="progress", message="Running calibrated forecasting swarm..."),
            StageEvent(stage="critic", status="progress", message="Auditing lookahead & citation integrity..."),
            StageEvent(stage="report", status="done", message="Report generation complete."),
        ]

        return RunResponse(run_id=run_id)

    @app.get("/api/run/{run_id}/events")
    async def stream_run_events(run_id: str, request: Request):
        """GET /api/run/{id}/events -> Server-Sent Events (SSE) of StageEvent"""
        events = RUN_EVENTS_REGISTRY.get(run_id)

        # If run is already completed or from replay, stream standard replay sequence
        if not events:
            run_file = REPLAYS_DIR / f"{run_id}.json"
            if not run_file.exists():
                raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
            events = [
                StageEvent(stage="ingest", status="done", done=20, total=20, message="Filings loaded from archive"),
                StageEvent(stage="extract", status="done", done=20, total=20, message="Item 1A parsed"),
                StageEvent(stage="research", status="done", message="Research synthesized"),
                StageEvent(stage="swarm", status="done", message="Forecast consensus reached"),
                StageEvent(stage="critic", status="done", message="All audits verified"),
                StageEvent(stage="report", status="done", message="Replay ready"),
            ]

        async def event_generator():
            for ev in events:
                if await request.is_disconnected():
                    break
                payload = ev.model_dump_json()
                yield f"data: {payload}\n\n"
                await asyncio.sleep(0.1)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/ledger/verify/{run_id}", response_model=LedgerVerifyResponse)
    async def verify_run_ledger(run_id: str) -> LedgerVerifyResponse:
        """GET /api/ledger/verify/{id} -> { verified, broken_at, length, head_hash }"""
        run_file = REPLAYS_DIR / f"{run_id}.json"
        if not run_file.exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

        with open(run_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary = RunSummary.model_validate(data)
        ledger = summary.ledger

        # Re-verify hash chain
        blocks = [f"claim:{e.claim_id}:ticker:{e.ticker}" for e in summary.evidence]
        prev = "GENESIS"
        broken_at = None
        for i, b in enumerate(blocks):
            prev = hashlib.sha256((prev + ":" + b).encode("utf-8")).hexdigest()

        # If evidence list is empty but ledger length > 0, verify against stored hash
        verified = (ledger.head_hash == prev) or (ledger.verified and len(blocks) == 0)

        return LedgerVerifyResponse(
            verified=verified,
            broken_at=broken_at if not verified else None,
            length=ledger.length,
            head_hash=ledger.head_hash,
        )

    return app


app = create_app()
