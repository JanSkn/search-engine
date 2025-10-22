#!/bin/bash
set -euo pipefail

PIDS=()

pushd src/backend
uv run uvicorn api.v1.app:app --host 127.0.0.1 --port 8000 "$@" &
PIDS+=($!)
echo "Uvicorn server started with PID ${PIDS[0]}"
popd

pushd src/frontend
npm run dev &
PIDS+=($!)
echo "Vite server started with PID ${PIDS[1]}"
popd

cleanup() {
  echo "Stopping all services..."
  kill "${PIDS[@]}" 2>/dev/null || true
  wait "${PIDS[@]}" 2>/dev/null || true
}

trap cleanup EXIT
wait