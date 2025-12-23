#!/bin/bash
set -e

cd search_engine/scripts/ && \
    chmod +x build-index.sh && \
    ./build-index.sh 1024 -1

uv run pytest /app/src/backend/tests