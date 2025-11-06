#!/bin/bash
set -e

# if no args passed, use /app/tests as default
if [ $# -eq 0 ]; then
    exec uv run pytest /app/tests
else
    exec uv run pytest "$@"
fi