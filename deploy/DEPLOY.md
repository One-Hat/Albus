# Deployment Guide: ALBUS on Nebius Serverless AI Cloud

> **Notice**: Per project constraints, this document provides the step-by-step deployment instructions and official Nebius CLI / Console specifications. No live cloud deploy, authentication, or billing operations were executed by the agent.

Official Nebius Documentation Sources:
- Nebius Serverless Overview: https://docs.nebius.com/serverless/
- Serverless Endpoints: https://docs.nebius.com/serverless/endpoints/
- Serverless Jobs: https://docs.nebius.com/serverless/jobs/

---

## 1. Prerequisites and Container Preparation

1. **Install and Configure Nebius CLI**  
   Doc URL: https://docs.nebius.com/cli/  
   Initialize your identity and select your target parent project:
   ```bash
   nebius init
   nebius auth login
   ```

2. **Container Image Build & Push to Registry**  
   Doc URL: https://docs.nebius.com/container-registry/  
   Build the ALBUS production image locally:
   ```bash
   docker build -t <REGISTRY_HOST>/<PROJECT_ID>/albus:v0.1.0 .
   docker push <REGISTRY_HOST>/<PROJECT_ID>/albus:v0.1.0
   ```
   *(If pulling from a private registry, configure credentials in your Nebius project settings).*

---

## 2. Deploying ALBUS Interactive Copilot as a Nebius Serverless Endpoint

Nebius Serverless Endpoints provide long-running HTTP/HTTPS services with managed domain URLs for real-time applications and APIs.

Doc URL: https://docs.nebius.com/serverless/endpoints/

### Step-by-Step Instructions:

1. **Obtain Project ID and Secret Credentials**  
   Find your Nebius project ID (`parent-id`):
   ```bash
   nebius iam project list
   ```

2. **Create the Serverless Endpoint via CLI**  
   Doc URL: https://docs.nebius.com/serverless/endpoints/  
   Run the `nebius ai endpoint create` command with exposed container ports:
   ```bash
   nebius ai endpoint create \
     --name albus-copilot \
     --parent-id <PROJECT_ID> \
     --image <REGISTRY_HOST>/<PROJECT_ID>/albus:v0.1.0 \
     --container-port 8080 \
     --container-port 8501/http \
     --env SEC_USER_AGENT="ALBUS-Research/1.0 (research@example.com)" \
     --env NEBIUS_BASE_URL="https://api.tokenfactory.nebius.com/v1"
   ```
   *Notes on flags:*
   - `--container-port 8080`: Serves the `/health` endpoint probe.
   - `--container-port 8501/http`: Nebius automatically generates a managed HTTPS public URL (`https://<ENDPOINT_HOST>`) routing to the Streamlit UI without requiring a separate public IP address.
   - Secret injection flag for endpoints: `UNKNOWN` (Nebius docs currently document passing secrets via Web Console Secret Manager or container environment references).

3. **Alternative: Deploy via Nebius Web Console**  
   Doc URL: https://docs.nebius.com/serverless/endpoints/  
   - Navigate to **Serverless AI** > **Endpoints** in the web console.
   - Click **Create Endpoint** and set name to `albus-copilot`.
   - Under **Configuration**, choose **Custom**.
   - Input **Image path**: `<REGISTRY_HOST>/<PROJECT_ID>/albus:v0.1.0`.
   - Set **Ports**: `8080` (HTTP health check) and `8501` (HTTP UI).
   - Under **Environment variables**, set `SEC_USER_AGENT` and secret `NEBIUS_API_KEY`.
   - Click **Create**. Once status shows `Running`, copy the public HTTPS URL.

4. **Verify Endpoint Health**  
   Send a GET request to the managed URL:
   ```bash
   curl -f https://<ENDPOINT_HOST>/health
   ```
   Expected response:
   ```json
   {
     "status": "healthy",
     "service": "albus-quant-copilot",
     "mode": "LIVE"
   }
   ```

---

## 3. Deploying Batch Benchmarks & Swarm Extraction as a Nebius Serverless Job

Nebius Serverless Jobs run finite batch workloads to completion (e.g., executing `scripts/run_benchmarks.py` or full 50-filing bulk extraction), automatically releasing all compute resources upon termination.

Doc URL: https://docs.nebius.com/serverless/jobs/

### Step-by-Step Instructions:

1. **Configure Job Execution Command**  
   Doc URL: https://docs.nebius.com/serverless/jobs/  
   Create the job targeting the benchmark script:
   ```bash
   nebius ai job create \
     --name albus-benchmark-run-01 \
     --image <REGISTRY_HOST>/<PROJECT_ID>/albus:v0.1.0 \
     --platform gpu-l40s-a \
     --preset 1gpu-8vcpu-32gb \
     --container-command python \
     --args "scripts/run_benchmarks.py" \
     --env-secret NEBIUS_API_KEY=<SECRET_NAME_OR_KEY> \
     --env SEC_USER_AGENT="ALBUS-Research/1.0 (research@example.com)" \
     --timeout 30m \
     --restart-policy on-failure
   ```
   *Notes on flags (strictly verified against docs):*
   - `--platform`: Hardware platform type (`gpu-l40s-a` per official docs; pure CPU platform ID: `UNKNOWN`).
   - `--preset`: Resource size preset (e.g. `1gpu-8vcpu-32gb`).
   - `--container-command`: Overrides entrypoint (`python`).
   - `--args`: Arguments passed to command (`scripts/run_benchmarks.py`).
   - `--env-secret`: Passes referenced secret securely to the job environment.
   - `--timeout`: Maximum wall-clock runtime (`30m`).
   - `--restart-policy`: `on-failure`.

2. **Monitor Job Execution and Stream Logs**  
   Doc URL: https://docs.nebius.com/serverless/jobs/  
   Follow the real-time execution logs until completion:
   ```bash
   nebius ai logs <JOB_ID> --follow
   ```

3. **Check Job Status and Exit Code**  
   ```bash
   nebius ai job get --id <JOB_ID>
   ```
   Inspect status field (`COMPLETED` or `FAILED`).

---

## 4. Operational & Security Checklist

1. **No Hardcoded Secrets**: Ensure `NEBIUS_API_KEY` is never baked into Docker images or committed to git.
2. **SEC Fair Access**: `SEC_USER_AGENT` must follow the SEC declaration standard: `SampleCompany AdminContact@domain.com`.
3. **Banner Verification**: When running without live credentials, confirm the UI displays the `CACHED DEMO / MOCK MODE` banner at the top of the interface.
