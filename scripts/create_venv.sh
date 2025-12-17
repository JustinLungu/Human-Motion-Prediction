#!/usr/bin/env bash
set -euo pipefail

VENV_NAME="uci_project"

echo "Removing existing virtual environment (if any)..."
rm -rf "${VENV_NAME}"

echo "Creating virtual environment: ${VENV_NAME}"
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

python3 -m venv "${VENV_NAME}"

echo ""
echo "Virtual environment '${VENV_NAME}' created successfully."
echo "Installing dependencies from requirements.txt..."

"${VENV_NAME}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_NAME}/bin/python" -m pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "To start working, run:"
echo "  source ${VENV_NAME}/bin/activate"
