#!/usr/bin/env bash
set -e

CONFIG="${1:-configs/default.yaml}"

echo "========================================"
echo "Train ConvLSTM U-Net"
echo "Config: ${CONFIG}"
echo "========================================"

python scripts/train_convlstm_unet.py --config "$CONFIG"
