#!/usr/bin/env python
"""One-batch CAWFE-Latte training smoke test on the rebuilt processed dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from src.config import load_config
from src.data.dataset import create_dataloaders
from src.models.model_factory import build_model_from_config
from src.training.batch_utils import unpack_batch
from src.training.hardware import autocast_context, choose_amp_dtype
from src.training.losses import get_loss_function
from src.training.metrics import compute_metrics
from src.training.model_outputs import extract_aux_outputs, extract_prediction
from src.training.train import _ensure_config_path, _get_device, _infer_input_channels_from_loader


def _apply_overrides(config: dict[str, Any], batch_size: int, num_workers: int, device: str | None) -> dict[str, Any]:
    config = dict(config)
    config["return_metadata"] = False
    config["batch_size"] = int(batch_size)
    training = dict(config.get("training", {}) if isinstance(config.get("training"), Mapping) else {})
    training["batch_size"] = int(batch_size)
    training["num_workers"] = int(num_workers)
    if device not in (None, "", "null"):
        training["device"] = str(device)
    config["training"] = training
    data_loader = dict(config.get("data_loader", {}) if isinstance(config.get("data_loader"), Mapping) else {})
    data_loader["batch_size"] = int(batch_size)
    data_loader["num_workers"] = int(num_workers)
    for split in ("train", "val", "test"):
        split_cfg = dict(data_loader.get(split, {}) if isinstance(data_loader.get(split), Mapping) else {})
        split_cfg["batch_size"] = int(batch_size)
        split_cfg["num_workers"] = int(num_workers)
        split_cfg["persistent_workers"] = False
        data_loader[split] = split_cfg
    config["data_loader"] = data_loader
    return config


def _assert_batch(name: str, x: torch.Tensor, y: torch.Tensor, terrain: torch.Tensor | None) -> None:
    if x.ndim != 5:
        raise ValueError(f"{name}: expected x shape B,T,C,H,W; got {tuple(x.shape)}")
    if y.ndim != 4 or int(y.shape[1]) != 4:
        raise ValueError(f"{name}: expected y shape B,4,H,W; got {tuple(y.shape)}")
    if tuple(x.shape[-2:]) != tuple(y.shape[-2:]):
        raise ValueError(f"{name}: input/target spatial mismatch x={tuple(x.shape)} y={tuple(y.shape)}")
    if terrain is not None:
        if terrain.ndim != 4 or int(terrain.shape[1]) != 4 or tuple(terrain.shape[-2:]) != tuple(y.shape[-2:]):
            raise ValueError(f"{name}: expected terrain shape B,4,H,W aligned to y; got terrain={tuple(terrain.shape)} y={tuple(y.shape)}")
        if not torch.isfinite(terrain).all():
            raise ValueError(f"{name}: terrain contains NaN/Inf")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError(f"{name}: x/y contains NaN/Inf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one CAWFE-Latte train/val smoke step.")
    parser.add_argument("--config", default="configs/experiments/cawfe_latte_v1.yaml")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = _ensure_config_path(load_config(config_path), config_path)
    config = _apply_overrides(config, max(1, args.batch_size), max(0, args.num_workers), args.device)

    if str(config.get("model", {}).get("architecture", "")).lower() != "cawfe_latte":
        raise ValueError("This smoke test expects model.architecture=cawfe_latte.")
    if not bool(config.get("cawfe_latte", {}).get("use_terrain_conditioning", False)):
        raise ValueError("This smoke test expects cawfe_latte.use_terrain_conditioning=true.")

    train_loader, val_loader, _test_loader = create_dataloaders(config)
    input_channels = _infer_input_channels_from_loader(train_loader)
    device = _get_device(config)
    model = build_model_from_config(config, input_channels=input_channels).to(device)
    criterion = get_loss_function(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=0.0)
    amp_dtype = choose_amp_dtype(config, device)

    train_batch = next(iter(train_loader))
    x_train, y_train, extra_train = unpack_batch(train_batch)
    terrain_train = extra_train.get("terrain")
    _assert_batch("train", x_train, y_train, terrain_train)
    x_train = x_train.to(device)
    y_train = y_train.to(device).float()
    terrain_train = terrain_train.to(device) if terrain_train is not None else None
    if terrain_train is None:
        raise ValueError("Training batch did not include terrain, but CAWFE-Latte terrain conditioning is enabled.")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, amp_dtype):
        train_output = model(x_train, terrain=terrain_train)
        train_pred = extract_prediction(train_output)
        loss_result = criterion(train_output, y_train)
        train_loss = loss_result["total_loss"] if isinstance(loss_result, Mapping) else loss_result
    if not torch.isfinite(train_loss):
        raise ValueError("Train smoke loss is not finite.")
    train_loss.backward()
    optimizer.step()

    val_batch = next(iter(val_loader))
    x_val, y_val, extra_val = unpack_batch(val_batch)
    terrain_val = extra_val.get("terrain")
    _assert_batch("val", x_val, y_val, terrain_val)
    x_val = x_val.to(device)
    y_val = y_val.to(device).float()
    terrain_val = terrain_val.to(device) if terrain_val is not None else None
    if terrain_val is None:
        raise ValueError("Validation batch did not include terrain, but CAWFE-Latte terrain conditioning is enabled.")

    model.eval()
    with torch.inference_mode(), autocast_context(device, amp_dtype):
        val_output = model(x_val, terrain=terrain_val)
        val_pred = extract_prediction(val_output).float()
        val_loss_result = criterion(val_output, y_val)
        val_loss = val_loss_result["total_loss"] if isinstance(val_loss_result, Mapping) else val_loss_result
    metrics = compute_metrics(val_pred.detach().cpu(), y_val.detach().cpu(), config)

    summary = {
        "status": "OK",
        "config": str(config_path),
        "device": str(device),
        "input_channels": int(input_channels),
        "train_x_shape": tuple(x_train.shape),
        "train_y_shape": tuple(y_train.shape),
        "train_terrain_shape": tuple(terrain_train.shape),
        "train_prediction_shape": tuple(train_pred.shape),
        "train_loss": float(train_loss.detach().item()),
        "val_x_shape": tuple(x_val.shape),
        "val_y_shape": tuple(y_val.shape),
        "val_terrain_shape": tuple(terrain_val.shape),
        "val_prediction_shape": tuple(val_pred.shape),
        "val_loss": float(val_loss.detach().item()),
        "aux_keys": sorted(extract_aux_outputs(train_output).keys()),
        "metrics": {key: (None if not math.isfinite(float(value)) else float(value)) for key, value in metrics.items()},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
