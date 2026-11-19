#!/usr/bin/env bash
# Start the earlier deepwise_rag project as a sidecar HTTP service on port 8100.
# It serves BM25 retrieval over the prepared PMC-Patients subset (20,400 published case summaries).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/_ref/deepwise_rag/backend"
export INSPECT_RAG_APP_CONFIG=pmc_patients.yaml
export INSPECT_RAG_ARTIFACT_DIR=../artifacts/pmc_patients
export INSPECT_RAG_DATABASE_URL=sqlite:///../artifacts/pmc_patients/inspect_rag.db
exec "$ROOT/_ref/.venv-rag/Scripts/python" -m app.cli serve --port "${DWAGENT_RAG_PORT:-8100}"
