#!/bin/bash
set -e

just -f /app/justfile build-index

uv run pytest /app/tests