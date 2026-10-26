"""
Unit tests for Phase 10: Deploy on Nebius (Serverless Endpoints, Jobs, Dockerfile, Health Check).
Tests that deployment manifests, Dockerfile, health server, and offline fallback mode adhere to specs.
"""
from __future__ import annotations
import json
import os
from http.server import HTTPServer
from pathlib import Path
from threading import Thread
from typing import Generator
import urllib.request
import pytest
import yaml

from src.health_server import HealthHandler, run_health_server


@pytest.fixture
def run_test_health_server() -> Generator[str, None, None]:
    """Start the health server on an ephemeral loopback port for testing."""
    # Find free port or let OS bind
    server = HTTPServer(("127.0.0.1", 0), HealthHandler)
    host, port = server.server_address
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://{host}:{port}/health"
    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()


def test_dockerfile_and_dockerignore_structure():
    """Verify Dockerfile and .dockerignore contain the required specifications."""
    dockerfile_path = Path("Dockerfile")
    dockerignore_path = Path(".dockerignore")

    assert dockerfile_path.exists(), "Dockerfile must exist at repo root"
    assert dockerignore_path.exists(), ".dockerignore must exist at repo root"

    df_content = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in df_content or "FROM python:3." in df_content
    assert "EXPOSE 8080" in df_content, "Must expose health check port 8080"
    assert "EXPOSE 8501" in df_content, "Must expose Streamlit UI port 8501"
    assert "HEALTHCHECK" in df_content, "Must include container HEALTHCHECK directive"
    assert "src/health_server.py" in df_content, "Must launch health_server.py"
    assert "src/app.py" in df_content, "Must launch Streamlit src/app.py"

    di_content = dockerignore_path.read_text(encoding="utf-8")
    assert ".git" in di_content
    assert ".venv" in di_content
    assert ".env" in di_content


def test_endpoint_and_job_manifests():
    """Verify deploy/endpoint.yaml and deploy/job.yaml are valid YAML and match Nebius specs."""
    endpoint_path = Path("deploy/endpoint.yaml")
    job_path = Path("deploy/job.yaml")

    assert endpoint_path.exists(), "deploy/endpoint.yaml must exist"
    assert job_path.exists(), "deploy/job.yaml must exist"

    endpoint_yaml = yaml.safe_load(endpoint_path.read_text(encoding="utf-8"))
    assert endpoint_yaml["kind"] == "Endpoint"
    spec = endpoint_yaml["spec"]
    ports = [p["container_port"] for p in spec["ports"]]
    assert 8080 in ports, "Endpoint must expose port 8080 for health probes"
    assert 8501 in ports, "Endpoint must expose port 8501 for Streamlit UI"

    job_yaml = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    assert job_yaml["kind"] == "Job"
    jspec = job_yaml["spec"]
    assert jspec["platform"] == "gpu-l40s-a", "Job must target gpu-l40s-a per Nebius docs"
    assert "run_benchmarks.py" in " ".join(jspec["args"])
    assert jspec["restart_policy"] == "on-failure"


def test_deploy_docs_integrity():
    """Verify deploy/DEPLOY.md contains verified documentation URLs and numbered steps."""
    deploy_md = Path("deploy/DEPLOY.md")
    assert deploy_md.exists(), "deploy/DEPLOY.md must exist"
    text = deploy_md.read_text(encoding="utf-8")

    assert "https://docs.nebius.com/serverless/endpoints/" in text
    assert "https://docs.nebius.com/serverless/jobs/" in text
    assert "nebius ai endpoint create" in text
    assert "nebius ai job create" in text
    assert "UNKNOWN" in text, "Must explicitly denote UNKNOWN where Nebius docs do not specify flags"


def test_health_server_cached_demo_mode(run_test_health_server, monkeypatch):
    """Verify health server returns 200 and indicates CACHED DEMO / MOCK when key is missing."""
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    url = run_test_health_server

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert payload["status"] == "healthy"
    assert payload["service"] == "albus-quant-copilot"
    assert payload["mode"] == "CACHED DEMO / MOCK"
    assert payload["live_api_key_configured"] is False


def test_health_server_live_mode(run_test_health_server, monkeypatch):
    """Verify health server returns 200 and indicates LIVE when key is present."""
    monkeypatch.setenv("NEBIUS_API_KEY", "synthetic-test-key-live-1234")
    url = run_test_health_server

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert payload["status"] == "healthy"
    assert payload["mode"] == "LIVE"
    assert payload["live_api_key_configured"] is True
