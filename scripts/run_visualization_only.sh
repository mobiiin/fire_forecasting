#!/usr/bin/env bash
set -e

CONFIG="${1:-configs/default.yaml}"
NUM_SAMPLES="${2:-10}"

echo "========================================"
echo "Visualize ConvLSTM U-Net Predictions"
echo "Config: ${CONFIG}"
echo "Num samples: ${NUM_SAMPLES}"
echo "========================================"

python scripts/visualize_predictions.py --config "$CONFIG" --num_samples "$NUM_SAMPLES"
