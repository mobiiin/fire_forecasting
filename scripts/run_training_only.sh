#!/usr/bin/env bash
set -e

CONFIG="${1:-configs/default.yaml}"

echo "========================================"
echo "Train ConvLSTM U-Net"
echo "Config: ${CONFIG}"
echo "========================================"

python scripts/precompute_patch_cache.py --config "$CONFIG" --split train
python scripts/compute_normalization.py --config "$CONFIG" --from_cache
python scripts/precompute_patch_cache.py --config "$CONFIG" --split val
python scripts/precompute_patch_cache.py --config "$CONFIG" --split test
python scripts/inspect_patch_cache.py --config "$CONFIG" --split all
python scripts/train_convlstm_unet.py --config "$CONFIG"
