#!/bin/bash
set -euo pipefail

export LOG_LEVEL=DEBUG

PIDS=()

cd src
uv run --project backend --refresh uvicorn backend.api.v1.app:app --host 127.0.0.1 --port 8000 "$@" &
PIDS+=($!)
echo "Uvicorn server started with PID ${PIDS[0]}"

cd frontend
npm run dev &
PIDS+=($!)
echo "Vite server started with PID ${PIDS[1]}"

cleanup() {
  echo "Stopping all services..."
  kill "${PIDS[@]}" 2>/dev/null || true
  wait "${PIDS[@]}" 2>/dev/null || true
}

trap cleanup EXIT
wait