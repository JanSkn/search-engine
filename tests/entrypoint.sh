#!/bin/bash
set -e

# if no args passed, use /app/tests as default
if [ $# -eq 0 ]; then
    exec uv run pytest /app/tests --cov=backend --cov-report=html:../../tests/htmlcov ../../tests/ # TODO report unnecessary for github workflow
else
    exec uv run pytest "$@"
fi