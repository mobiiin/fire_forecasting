"""Sanity-check the wildfire forecasting project before training."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import torch  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
    torch = None
    DataLoader = None

from src.config import load_config
from src.data.dataset import FireSequenceDataset, _sort_chronologically
from src.data.preprocessing import load_normalization_stats
from src.data.splits import chronological_split_indices
from src.models.convlstm_unet import build_model_from_config
from src.training.losses import get_loss_function


def _resolve_path(base_path: Path, configured_path: str | Path) -> Path:
    """Resolve config-relative paths."""

    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_path.parent / path).resolve()


def _print_environment_info() -> None:
    """Print Python/PyTorch/CUDA environment details."""

    print("Environment")
    print(f"  Python: {platform.python_version()}")
    if torch is None:
        print("  PyTorch: not installed")
        print("  CUDA available: False")
        return

    print(f"  PyTorch: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")


def _validate_required_keys(config: Mapping[str, Any], keys: list[str]) -> None:
    """Validate required top-level config keys."""

    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(f"Missing required config key(s): {', '.join(missing)}")


def _discover_files(config: Mapping[str, Any], config_path: Path) -> list[Path]:
    """Discover and chronologically sort dataset files."""

    data_dir = _resolve_path(config_path, str(config["data_dir"]))
    file_pattern = str(config["file_pattern"])
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = _sort_chronologically(list(data_dir.glob(file_pattern)))
    if not files:
        raise FileNotFoundError(f"No files found in '{data_dir}' using pattern '{file_pattern}'.")

    print(f"Discovered {len(files)} files in {data_dir}")
    return files


def _print_first_tensor_stats(files: list[Path]) -> tuple[int, int, int]:
    """Load first tensor and print shape/statistics."""

    first_tensor = np.load(files[0], allow_pickle=False)
    if first_tensor.ndim != 3:
        raise ValueError(
            f"Expected first tensor rank to be 3 ((H, W, C)), got shape {first_tensor.shape} at {files[0]}"
        )

    h, w, c = first_tensor.shape
    finite_mask = np.isfinite(first_tensor)
    finite_values = first_tensor[finite_mask]
    if finite_values.size == 0:
        raise ValueError(f"First tensor has no finite values: {files[0]}")

    print("First tensor stats")
    print(f"  path: {files[0]}")
    print(f"  shape: {first_tensor.shape}")
    print(f"  dtype: {first_tensor.dtype}")
    print(f"  min: {float(finite_values.min()):.6g}")
    print(f"  max: {float(finite_values.max()):.6g}")
    print(f"  mean: {float(finite_values.mean()):.6g}")
    print(f"  std: {float(finite_values.std()):.6g}")
    print(f"  NaN count: {int(np.isnan(first_tensor).sum())}")
    print(f"  Inf count: {int(np.isinf(first_tensor).sum())}")

    return int(h), int(w), int(c)


def _build_datasets(config: Mapping[str, Any], files: list[Path], normalization_stats) -> tuple[FireSequenceDataset, FireSequenceDataset, FireSequenceDataset]:
    """Create train/val/test datasets with chronological splits."""

    input_sequence_length = int(config["input_sequence_length"])
    prediction_horizon = int(config["prediction_horizon"])
    input_channel_count = int(config.get("input_channel_count", config.get("model", {}).get("input_channels", 0)))
    if input_channel_count <= 0:
        raise KeyError("Config must define a positive input_channel_count or model.input_channels.")
    split_indices = chronological_split_indices(
        num_timesteps=len(files),
        input_sequence_length=input_sequence_length,
        prediction_horizon=prediction_horizon,
        train_fraction=float(config.get("train_fraction", 0.7)),
        val_fraction=float(config.get("val_fraction", 0.15)),
        test_fraction=float(config.get("test_fraction", 0.15)),
    )

    print("Chronological sample counts")
    print(f"  train: {len(split_indices['train'])}")
    print(f"  val:   {len(split_indices['val'])}")
    print(f"  test:  {len(split_indices['test'])}")

    common_kwargs = {
        "file_paths": files,
        "input_sequence_length": input_sequence_length,
        "prediction_horizon": prediction_horizon,
        "target_channel": int(config["target_channel"]),
        "input_channel_count": input_channel_count,
        "task_type": str(config.get("task_type", "regression")),
        "fire_threshold": float(config.get("fire_threshold", 0.5)),
        "normalization_stats": normalization_stats,
        "patch_size": int(config.get("patch_size", 64)),
        "active_patch_probability": float(config.get("active_patch_probability", 0.7)),
        "active_threshold": float(config.get("active_threshold", config.get("fire_threshold", 0.5))),
    }

    train_dataset = FireSequenceDataset(
        sample_indices=split_indices["train"],
        use_patches=bool(config.get("use_patches", False)),
        **common_kwargs,
    )
    val_dataset = FireSequenceDataset(
        sample_indices=split_indices["val"],
        use_patches=bool(config.get("use_patches_for_eval", False)),
        **common_kwargs,
    )
    test_dataset = FireSequenceDataset(
        sample_indices=split_indices["test"],
        use_patches=bool(config.get("use_patches_for_eval", False)),
        **common_kwargs,
    )

    return train_dataset, val_dataset, test_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description="Sanity-check the wildfire forecasting project.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    return parser


def main() -> None:
    """Run project sanity checks end-to-end."""

    args = build_arg_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)

    _print_environment_info()
    if torch is None or DataLoader is None:
        raise ImportError(
            "PyTorch is required for sanity_check_project.py. Install torch and retry."
        )

    _validate_required_keys(
        config,
        ["data_dir", "file_pattern", "input_sequence_length", "prediction_horizon", "target_channel"],
    )

    files = _discover_files(config, config_path)
    _, _, c = _print_first_tensor_stats(files)

    target_channel = int(config["target_channel"])
    if target_channel < 0 or target_channel >= c:
        raise ValueError(
            f"target_channel out of range: got {target_channel}, but channel count is {c}."
        )

    normalization_stats = None
    normalization_cfg = config.get("normalization", {})
    normalization_path = normalization_cfg.get("path")
    if normalization_path:
        stats_path = _resolve_path(config_path, normalization_path)
        if stats_path.exists():
            normalization_stats = load_normalization_stats(stats_path)
            print(f"Loaded normalization stats: {stats_path}")
        else:
            print("Warning: normalization stats not found.")
            print("Run: python scripts/compute_normalization.py --config configs/default.yaml")

    train_dataset, val_dataset, test_dataset = _build_datasets(config, files, normalization_stats)

    batch_size = int(config.get("batch_size", 4))
    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    if len(train_loader.dataset) == 0:
        raise ValueError("Train dataset is empty; cannot run sanity forward pass.")

    x_batch, y_batch = next(iter(train_loader))[:2]
    if x_batch.ndim != 5:
        raise ValueError(f"Expected X batch shape (B, T, C, H, W), got {tuple(x_batch.shape)}")
    if y_batch.ndim != 4:
        raise ValueError(f"Expected y batch shape (B, 1, H, W), got {tuple(y_batch.shape)}")
    if y_batch.shape[1] != 1:
        raise ValueError(f"Expected y channel dimension 1, got {y_batch.shape[1]}")

    print(f"Batch shapes: X={tuple(x_batch.shape)}, y={tuple(y_batch.shape)}")
    print(f"Dataset lengths: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    device_name = str(config.get("device", "auto")).lower()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = build_model_from_config(config, input_channels=int(x_batch.shape[2])).to(device)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        y_pred = model(x_batch)

    if y_pred.shape != y_batch.shape:
        raise ValueError(
            f"Output shape mismatch: expected {tuple(y_batch.shape)}, got {tuple(y_pred.shape)}"
        )

    criterion = get_loss_function(config)
    loss_value = criterion(y_pred, y_batch)
    if not torch.isfinite(loss_value):
        raise ValueError(f"Loss is non-finite: {float(loss_value.item())}")

    print(f"Single forward pass loss: {float(loss_value.item()):.6f}")
    print("Sanity check passed")


if __name__ == "__main__":
    main()
