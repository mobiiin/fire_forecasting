#!/usr/bin/env bash
set -e

CONFIG="${1:-configs/default.yaml}"

echo "========================================"
echo "Wildfire Forecasting Full Pipeline"
echo "Config: ${CONFIG}"
echo "========================================"

echo
echo "[1/8] Inspect dataset"
python scripts/inspect_dataset.py --config "$CONFIG"

echo
echo "[2/8] Precompute train patch cache"
python scripts/precompute_patch_cache.py --config "$CONFIG" --split train

echo
echo "[3/8] Compute normalization stats from train cache"
python scripts/compute_normalization.py --config "$CONFIG" --from_cache

echo
echo "[4/8] Precompute validation/test patch cache"
python scripts/precompute_patch_cache.py --config "$CONFIG" --split val
python scripts/precompute_patch_cache.py --config "$CONFIG" --split test

echo
echo "[5/8] Inspect patch cache"
python scripts/inspect_patch_cache.py --config "$CONFIG" --split all

echo
echo "[6/8] Project sanity check"
python scripts/sanity_check_project.py --config "$CONFIG"

echo
echo "[7/8] Train forecasting model"
python scripts/train_forecasting_model.py --config "$CONFIG"

echo
echo "[8/8] Visualize predictions"
python scripts/visualize_predictions.py --config "$CONFIG" --num_samples 10

echo
echo "Done. Rollout is intentionally skipped in full pipeline by default."
