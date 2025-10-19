#!/bin/bash

PIDS=()

poetry run uvicorn src.backend.api.v1.app:app --host 127.0.0.1 --port 8000 "$@" &
PIDS+=($!)
echo "Uvicorn server started with PID ${PIDS[-1]}"

npm run dev --prefix src/frontend &
PIDS+=($!)
echo "Vite server started with PID ${PIDS[-1]}"

cleanup() {
  echo "Stopping all services..."
  kill "${PIDS[@]}" 2>/dev/null || true
  wait "${PIDS[@]}" 2>/dev/null || true
}

trap cleanup EXIT