#!/bin/bash
set -euo pipefail

echo "Building inverted index..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$SCRIPT_DIR/../index_builder"
mkdir -p build
cd build
cmake ..
cmake --build .

./index_builder "$@"

./merge_partial_indices

echo "Creating vector embeddings..."
export LOG_LEVEL=DEBUG
cd "$SRC_DIR"
uv run --project backend python -m backend.search_engine.index_builder.create_embeddings "$2"
