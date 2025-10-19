#!/usr/bin/env bash
set -euo pipefail

echo "Installing Poetry..."
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"

echo "Poetry version: $(poetry --version)"

echo "Installing Just..."
sudo apt-get update -y
sudo apt-get install -y just

echo "Just version: $(just --version)"