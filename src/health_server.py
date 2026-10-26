"""
Lightweight HTTP Health Check Server for ALBUS Container & Kubernetes / Nebius Health Probes.
Serves GET /health and GET /healthz returning JSON status and active mode (LIVE vs MOCK / CACHED DEMO).
"""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Optional


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/health", "/healthz", "/"):
            is_live = bool(
                os.environ.get("NEBIUS_API_KEY", "").strip()
                or os.environ.get("GROQ_API_KEY", "").strip()
                or os.environ.get("OPENROUTER_API_KEY", "").strip()
            )
            payload = {
                "status": "healthy",
                "service": "albus-quant-copilot",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "LIVE" if is_live else "CACHED DEMO / MOCK",
                "live_api_key_configured": is_live,
                "version": "0.1.0",
                "endpoints": {
                    "health": "/health",
                    "ui": "http://localhost:8501",
                },
            }
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format: str, *args: object) -> None:
        # Suppress noisy health probe logging to stdout
        pass


def run_health_server(port: int = 8080) -> HTTPServer:
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"[INFO] ALBUS Health Server listening on http://0.0.0.0:{port}/health")
    return server


def start_health_server_in_background(port: int = 8080) -> HTTPServer:
    server = run_health_server(port=port)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    server = run_health_server(port=port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
        print("\n[INFO] Health server stopped.")
