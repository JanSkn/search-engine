#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv run --project "../$SCRIPT_DIR" pybind11-stubgen cpp_utils -o .      