"""Sanity-check the wildfire forecasting project before training."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None

from src.config import load_config
from src.data.dataset import _count_engineered_channels, _resolve_multitask_config, create_dataloaders
from src.models.convlstm_unet import build_model_from_config
from src.training.losses import get_loss_function


def _print_environment_info() -> None:
	"""Print Python/PyTorch/CUDA environment details."""

	print("Environment")
	print(f"  Python: {platform.python_version()}")
	if torch is None:
		print("  PyTorch: not installed")
		print("  CUDA available: False")
		return
	print(f"  PyTorch: {torch.__version__}")
	print(f"  CUDA available: {torch.cuda.is_available()}")
	if torch.cuda.is_available():
		print(f"  CUDA device: {torch.cuda.get_device_name(0)}")


def _tensor_stats(array_like) -> dict[str, float]:
	"""Compute min/max/mean/std for a tensor-like value."""

	if torch is not None and torch.is_tensor(array_like):
		array = array_like.detach().cpu().to(torch.float32).numpy()
	else:
		array = np.asarray(array_like, dtype=np.float32)
	finite_values = array[np.isfinite(array)]
	if finite_values.size == 0:
		return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
	return {
		"min": float(finite_values.min()),
		"max": float(finite_values.max()),
		"mean": float(finite_values.mean()),
		"std": float(finite_values.std()),
	}


def _format_stats(label: str, stats: Mapping[str, float]) -> str:
	"""Format a stats dictionary consistently."""

	return (
		f"{label}: min={stats['min']:.6g} max={stats['max']:.6g} "
		f"mean={stats['mean']:.6g} std={stats['std']:.6g}"
	)


def build_arg_parser() -> argparse.ArgumentParser:
	"""Build CLI parser."""

	parser = argparse.ArgumentParser(description="Sanity-check the wildfire forecasting project.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
	return parser


def main() -> None:
	"""Run project sanity checks end-to-end."""

	args = build_arg_parser().parse_args()
	if torch is None:
		raise ImportError("PyTorch is required for sanity_check_project.py.")
	config_path = Path(args.config).expanduser().resolve()
	config = load_config(config_path)
	config["config_path"] = str(config_path)

	_print_environment_info()
	train_loader, _, _ = create_dataloaders(config)
	if len(train_loader.dataset) == 0:
		raise ValueError("Train dataset is empty; cannot run sanity checks.")

	first_file = train_loader.dataset.file_paths[0]
	first_tensor = np.load(first_file, allow_pickle=False)
	if first_tensor.shape != (144, 144, 86):
		raise ValueError(f"Expected one raw tensor shaped (144, 144, 86), got {first_tensor.shape} at {first_file}.")
	print("Raw data")
	print(f"  path: {first_file}")
	print(f"  shape: {first_tensor.shape}")

	x_batch, y_batch = next(iter(train_loader))[:2]
	if x_batch.ndim != 5:
		raise ValueError(f"Expected X batch shape (B, T, C, H, W), got {tuple(x_batch.shape)}")
	if y_batch.ndim != 4:
		raise ValueError(f"Expected y batch shape (B, C, H, W), got {tuple(y_batch.shape)}")

	task_type = str(config.get("task_type", "regression")).lower()
	model = build_model_from_config(config, input_channels=int(x_batch.shape[2]))
	device_name = str(config.get("device", "auto")).lower()
	if device_name == "auto":
		device_name = "cuda" if torch.cuda.is_available() else "cpu"
	if device_name == "cuda" and not torch.cuda.is_available():
		device_name = "cpu"
	device = torch.device(device_name)
	model = model.to(device)
	x_batch = x_batch.to(device)
	y_batch = y_batch.to(device)

	with torch.no_grad():
		y_pred = model(x_batch)

	criterion = get_loss_function(config)
	loss_result = criterion(y_pred, y_batch)
	total_loss = loss_result["total_loss"] if isinstance(loss_result, dict) else loss_result
	if not torch.isfinite(total_loss):
		raise ValueError(f"Loss is non-finite: {float(total_loss.item())}")

	engineered_channel_count = _count_engineered_channels(config)
	print("Channels")
	print(f"  base input channels: {int(config.get('input_channel_count', 0))}")
	print(f"  engineered channels: {engineered_channel_count}")
	print(f"  total input channels: {int(x_batch.shape[2])}")

	if task_type == "multitask":
		multitask = _resolve_multitask_config(config)
		print(f"  surface_fuel_channel: {multitask['surface_fuel_channel']}")
		print(f"  canopy_fuel_channel: {multitask['canopy_fuel_channel']}")
		print(f"  flux_mask_channel: {multitask['flux_mask_channel']}")
		print(f"  mask_target_type: {multitask['mask_target_type']}")
		if multitask["mask_target_type"] == "active_flux":
			print("  channel 2 label: mask = future flux channel > flux_fire_threshold")
		else:
			print("  channel 2 label: mask = max(initial fuel - future surface/canopy fuel) > consumed_fuel_threshold")

	expected_y_channels = 3 if task_type == "multitask" else 1
	if y_batch.shape[1] != expected_y_channels:
		raise ValueError(f"Expected y channel dimension {expected_y_channels}, got {y_batch.shape[1]}")
	if tuple(y_pred.shape) != tuple(y_batch.shape):
		raise ValueError(f"Expected model output shape {tuple(y_batch.shape)}, got {tuple(y_pred.shape)}")

	print("Shapes")
	print(f"  X batch shape: {tuple(x_batch.shape)}")
	print(f"  y batch shape: {tuple(y_batch.shape)}")
	print(f"  model output shape: {tuple(y_pred.shape)}")

	if task_type == "multitask":
		surface_stats = _tensor_stats(y_batch[:, 0])
		canopy_stats = _tensor_stats(y_batch[:, 1])
		mask_values = torch.unique(y_batch[:, 2]).detach().cpu().numpy()
		if not np.all(np.isin(mask_values, np.asarray([0.0, 1.0], dtype=np.float32))):
			raise ValueError(f"Multitask mask contains values other than 0 and 1: {mask_values}")
		print(_format_stats("  y[:, 0] surface consumed fuel", surface_stats))
		print(_format_stats("  y[:, 1] canopy consumed fuel", canopy_stats))
		print(f"  y[:, 2] unique values: {mask_values.tolist()}")
		print(f"  y[:, 2] active pixel fraction: {float(y_batch[:, 2].float().mean().item()):.6f}")
	else:
		print(_format_stats("  y batch", _tensor_stats(y_batch)))

	print(f"Finite total loss: {float(total_loss.item()):.6f}")
	print("Sanity check passed")


if __name__ == "__main__":
	main()
