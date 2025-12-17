#!/usr/bin/env bash
set -euo pipefail

echo "Installing dependencies..."
python -m pip install --quiet gdown pyyaml

echo "Populating dataset..."
python src/populate_repo/get_data.py

echo "Done."
