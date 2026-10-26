#!/usr/bin/env bash
# Deploy ALBUS Interactive Copilot as a Nebius Serverless Endpoint
# Docs: https://docs.nebius.com/serverless/endpoints/
set -euo pipefail

PROJECT_ID="${NEBIUS_PROJECT_ID:?Error: NEBIUS_PROJECT_ID environment variable required}"
IMAGE="${CONTAINER_REGISTRY_IMAGE:-cr.nebius.cloud/albus:v0.1.0}"
USER_AGENT="${SEC_USER_AGENT:-ALBUS-Research/1.0 (research@example.com)}"

echo "[INFO] Creating Nebius Serverless Endpoint: albus-copilot..."
nebius ai endpoint create \
  --name albus-copilot \
  --parent-id "${PROJECT_ID}" \
  --image "${IMAGE}" \
  --container-port 8080 \
  --container-port 8501/http \
  --env SEC_USER_AGENT="${USER_AGENT}" \
  --env NEBIUS_BASE_URL="https://api.tokenfactory.nebius.com/v1"

echo "[INFO] Endpoint deployment initiated. Check status with:"
echo "nebius ai endpoint get-by-name --name albus-copilot"
