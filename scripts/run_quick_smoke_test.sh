#!/usr/bin/env bash
set -e

CONFIG="${1:-configs/default.yaml}"

echo "========================================"
echo "Wildfire Forecasting Quick Smoke Test"
echo "Config: ${CONFIG}"
echo "========================================"

echo
echo "[1/2] Inspect dataset"
python scripts/inspect_dataset.py --config "$CONFIG"

echo
echo "[2/2] Project sanity check"
python scripts/sanity_check_project.py --config "$CONFIG"

echo
echo "Quick smoke test passed."
