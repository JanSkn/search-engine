#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../index_builder"
mkdir -p build
cd build
cmake ..
cmake --build .

./index_builder "$@"