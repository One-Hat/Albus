# Contracts (Interface Boundary)

This directory contains the immutable interface contract between the **Albus Backend** (Python research pipeline, FastAPI, replays) and the **Albus Frontend** (`web/` client).

## Current Version
- `contract_version`: `1.0` (declared in `contracts/types.ts`)

## Ground Rules for Changes

1. **Mutual Approval**: The contract can change **ONLY** via a Pull Request touching `contracts/` that is reviewed and approved by both backend and frontend owners.
2. **Version Bump**: Every PR modifying `contracts/types.ts` **MUST** bump `contract_version` (e.g. `'1.0'` -> `'1.1'`).
3. **Drift Enforcement**: The backend maintains a strict Pydantic mirror (`src/api/models.py`) validated by automated drift tests (`tests/test_contracts_drift.py`). Any field addition, deletion, or renaming without updating both sides will immediately fail CI.
4. **No Ad-Hoc Fields**: Neither backend nor frontend may introduce uncontracted payload shapes into `/api/*` or `/replays/*.json`.
