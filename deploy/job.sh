#!/usr/bin/env bash
# Deploy Batch Benchmark Run as a Nebius Serverless Job
# Docs: https://docs.nebius.com/serverless/jobs/
set -euo pipefail

PROJECT_ID="${NEBIUS_PROJECT_ID:?Error: NEBIUS_PROJECT_ID environment variable required}"
IMAGE="${CONTAINER_REGISTRY_IMAGE:-cr.nebius.cloud/albus:v0.1.0}"
USER_AGENT="${SEC_USER_AGENT:-ALBUS-Research/1.0 (research@example.com)}"

echo "[INFO] Creating Nebius Serverless Job: albus-benchmark-run..."
nebius ai job create \
  --name "albus-benchmark-$(date +%s)" \
  --parent-id "${PROJECT_ID}" \
  --image "${IMAGE}" \
  --platform gpu-l40s-a \
  --preset 1gpu-8vcpu-32gb \
  --container-command python \
  --args "scripts/run_benchmarks.py" \
  --env-secret NEBIUS_API_KEY=nebius-token-factory-key \
  --env SEC_USER_AGENT="${USER_AGENT}" \
  --timeout 30m \
  --restart-policy on-failure

echo "[INFO] Job submitted. Monitor execution with:"
echo "nebius ai job list --parent-id ${PROJECT_ID}"
