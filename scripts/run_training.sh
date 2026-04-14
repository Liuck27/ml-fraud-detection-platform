#!/usr/bin/env bash
# Run both training scripts in order.  Exits immediately on any error.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Detect correct venv Python path (Windows Git Bash vs Linux/Mac)
if [[ -f "${REPO_ROOT}/training/.venv/Scripts/python" ]]; then
    PYTHON="${REPO_ROOT}/training/.venv/Scripts/python"
else
    PYTHON="${REPO_ROOT}/training/.venv/bin/python"
fi

echo "==> Using Python: ${PYTHON}"
echo "==> Training XGBoost classifier…"
"${PYTHON}" "${REPO_ROOT}/training/train_xgboost.py"

echo ""
echo "==> Training Autoencoder…"
"${PYTHON}" "${REPO_ROOT}/training/train_autoencoder.py"

echo ""
echo "==> Training complete."
