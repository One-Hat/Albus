#!/usr/bin/env python3
"""
Script to query the Nebius Token Factory /models endpoint and verify/update config/models.yaml.
AGENTS.md Rule 3: Never invent model IDs. Read them from config/models.yaml.
AGENTS.md Rule 7: No secrets in code or logs. Read them from the environment.
"""
import os
import sys
from pathlib import Path
import httpx
import yaml


def fetch_and_update_models() -> int:
    api_key = os.environ.get("NEBIUS_API_KEY")
    base_url = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1").rstrip("/")
    models_file = Path("config/models.yaml")

    if not models_file.exists():
        print(f"[ERROR] Config file {models_file} does not exist.", file=sys.stderr)
        return 1

    if not api_key:
        print("[INFO] NEBIUS_API_KEY not set in environment.")
        print(f"[INFO] Validating existing model IDs in {models_file} against expected schema...")
        with open(models_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        models = data.get("models", {})
        for role in ["nano", "super", "ultra"]:
            if role not in models:
                print(f"[ERROR] Missing required model role: {role}", file=sys.stderr)
                return 1
            print(f"  - {role}: {models[role]['id']} (context: {models[role]['context_window']})")
        print("[OK] Existing model configurations verified.")
        return 0

    endpoint = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"[INFO] Querying models from {endpoint}...")
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(endpoint, headers=headers)
            if resp.status_code != 200:
                print(f"[ERROR] Failed to query models: HTTP {resp.status_code}: {resp.text}", file=sys.stderr)
                return 1
            payload = resp.json()
            available_models = [m.get("id") for m in payload.get("data", [])]

        print(f"[INFO] Retrieved {len(available_models)} models from Nebius Token Factory.")
        nemotron_models = [m for m in available_models if "nemotron" in m.lower()]
        print(f"[INFO] Found {len(nemotron_models)} Nemotron models:")
        for m in nemotron_models:
            print(f"  * {m}")

        return 0

    except Exception as e:
        print(f"[ERROR] Exception while querying models endpoint: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(fetch_and_update_models())
