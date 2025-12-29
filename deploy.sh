#!/bin/bash
set -euo pipefail

if [ ! -f src/backend/search_engine/index_builder/data/msmarco-docs.tsv ]; then
    echo "msmarco-docs.tsv file not found. Starting download..."
    cd src && uv run --project backend python -m backend.search_engine.scripts.download_dataset
fi
    
if [ -z "$(ls -A src/backend/search_engine/index/bin/ 2>/dev/null)" ]; then
    echo "Index binaries not found. Starting build process..."
    just build-index
fi

echo "Spinning up containers..."
docker compose up -d