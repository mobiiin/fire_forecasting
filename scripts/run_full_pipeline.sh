#!/usr/bin/env bash
set -e

CONFIG="${1:-configs/default.yaml}"

echo "========================================"
echo "Wildfire Forecasting Full Pipeline"
echo "Config: ${CONFIG}"
echo "========================================"

echo
echo "[1/5] Inspect dataset"
python scripts/inspect_dataset.py --config "$CONFIG"

echo
echo "[2/5] Compute normalization stats"
python scripts/compute_normalization.py --config "$CONFIG"

echo
echo "[3/5] Project sanity check"
python scripts/sanity_check_project.py --config "$CONFIG"

echo
echo "[4/5] Train ConvLSTM U-Net"
python scripts/train_convlstm_unet.py --config "$CONFIG"

echo
echo "[5/5] Visualize predictions"
python scripts/visualize_predictions.py --config "$CONFIG" --num_samples 10

echo
echo "Done. Rollout is intentionally skipped in full pipeline by default."
