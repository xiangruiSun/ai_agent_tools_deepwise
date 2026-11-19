#!/usr/bin/env bash
# Start the whole on-site stack on one machine: retrieval sidecar, DICOM gateway, portal.
# Ollama must already be running with the configured model (default qwen3:14b).
#   bash scripts/start_all.sh                    start everything, Ctrl-C stops everything
#   DWAGENT_PUBLIC=1 bash scripts/start_all.sh   the same with the portal in public mode (see README)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/Scripts/python"
[ -x "$PY" ] || PY="$ROOT/.venv/bin/python"
PORT="${DWAGENT_PORT:-8200}"
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

if ! curl -s -m 3 "${DWAGENT_OLLAMA_URL:-http://127.0.0.1:11434}/api/tags" >/dev/null; then
  echo "Ollama is not reachable. Start it first: ollama serve" >&2; exit 1
fi
if [ -d "$ROOT/_ref/deepwise_rag" ]; then
  bash scripts/start_rag.sh > data/runtime/rag.log 2>&1 & pids+=($!)
  echo "retrieval sidecar  -> http://127.0.0.1:${DWAGENT_RAG_PORT:-8100}  (log: data/runtime/rag.log)"
else
  echo "retrieval sidecar  -> not installed; similar-case search will report itself unavailable"
fi
"$PY" -m dwagent.gateway.scp --port "${DWAGENT_DICOM_PORT:-11112}" > data/runtime/gateway.log 2>&1 & pids+=($!)
echo "DICOM gateway      -> 127.0.0.1:${DWAGENT_DICOM_PORT:-11112}  AE title DEEPWISE_AI"
echo "portal             -> http://127.0.0.1:${PORT}"
if [ "${DWAGENT_PUBLIC:-0}" = "1" ]; then      # behind a tunnel: generated secret and passwords, limits, headers
  DWAGENT_PORT="$PORT" "$PY" -m dwagent.web.public
else
  "$PY" -m uvicorn dwagent.web.app:app --host "${DWAGENT_HOST:-127.0.0.1}" --port "$PORT"
fi
