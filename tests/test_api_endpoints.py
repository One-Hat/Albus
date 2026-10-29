"""Integration tests for FastAPI endpoints."""
import os
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app, ConfigResponse
from src.api.models import RunIndexItem, RunSummary, StageEvent


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_get_config(client):
    """Test GET /api/config returns verified provider, models, and limits."""
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()

    assert "provider" in data
    assert "models" in data
    assert "small" in data["models"]
    assert "mid" in data["models"]
    assert "large" in data["models"]
    assert "live_enabled" in data
    assert data["live_enabled"] is False  # Default is False
    assert "limits" in data
    assert data["limits"]["max_tickers"] == 20


def test_get_runs_index(client):
    """Test GET /api/runs returns list of runs including mock-sample."""
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1

    ids = [item["id"] for item in data]
    assert "mock-sample" in ids

    # Validates with Pydantic
    for item in data:
        RunIndexItem.model_validate(item)


def test_get_run_details(client):
    """Test GET /api/runs/mock-sample returns complete RunSummary."""
    resp = client.get("/api/runs/mock-sample")
    assert resp.status_code == 200
    data = resp.json()

    summary = RunSummary.model_validate(data)
    assert summary.id == "mock-sample"
    assert summary.source == "MOCK"
    assert summary.backtest.n_events == 35


def test_get_run_details_not_found(client):
    """Test GET /api/runs/unknown-id returns 404."""
    resp = client.get("/api/runs/non-existent-xyz-999")
    assert resp.status_code == 404


def test_post_run_disabled_by_default(client):
    """Test POST /api/run returns 403 when LIVE_ENABLED=false."""
    os.environ["LIVE_ENABLED"] = "false"
    resp = client.post(
        "/api/run",
        json={
            "hypothesis": "Test hypothesis about supply chains",
            "universe": ["AAPL", "MSFT"],
        },
    )
    assert resp.status_code == 403
    assert "disabled" in resp.json()["detail"].lower()


def test_post_run_with_invalid_tickers(client, monkeypatch):
    """Test POST /api/run with invalid ticker returns 400 Bad Request."""
    monkeypatch.setenv("LIVE_ENABLED", "true")
    resp = client.post(
        "/api/run",
        json={
            "hypothesis": "Valid hypothesis about semiconductor supply chain disruptions",
            "universe": ["AAPL", "FAKE_INVALID_TICKER"],
        },
    )
    assert resp.status_code == 400
    assert "allowlist" in resp.json()["detail"].lower()


def test_post_run_live_enabled_success(client, monkeypatch):
    """Test POST /api/run returns run_id when enabled and valid."""
    monkeypatch.setenv("LIVE_ENABLED", "true")
    resp = client.post(
        "/api/run",
        json={
            "hypothesis": "Disclosures regarding enterprise cyber attacks predict lower returns.",
            "universe": ["AAPL", "MSFT", "NVDA"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    assert data["run_id"].startswith("live-run-")


def test_sse_run_events_stream(client):
    """Test GET /api/run/{id}/events streams SSE formatted StageEvents."""
    resp = client.get("/api/run/mock-sample/events")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n\n")
    assert len(lines) >= 1
    for chunk in lines:
        if chunk.startswith("data: "):
            payload_str = chunk[len("data: "):]
            ev = StageEvent.model_validate_json(payload_str)
            assert ev.stage in ["ingest", "extract", "research", "swarm", "critic", "report"]


def test_ledger_verify_endpoint(client):
    """Test GET /api/ledger/verify/mock-sample returns verified status."""
    resp = client.get("/api/ledger/verify/mock-sample")
    assert resp.status_code == 200
    data = resp.json()
    assert data["verified"] is True
    assert "head_hash" in data
    assert data["length"] > 0
