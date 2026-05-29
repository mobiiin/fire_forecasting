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
from src.data.dataset import (
	_count_fuel_flux_engineered_channels,
	_resolve_path,
	_sort_chronologically,
	count_atmospheric_engineered_channels,
	_resolve_atmospheric_features_config,
	_resolve_multitask_config,
	create_dataloaders,
	resolve_engineered_feature_slices,
)
from src.data.spatial_transforms import infer_with_external_test_spatial_handling
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
	data_dir = _resolve_path(config_path, config["data_dir"])
	atmospheric = _resolve_atmospheric_features_config(config)
	raw_files = _sort_chronologically(list(data_dir.glob(str(config["file_pattern"]))))
	if not raw_files:
		raise FileNotFoundError(f"No files found in '{data_dir}' using pattern '{config['file_pattern']}'.")
	first_file = raw_files[0]
	first_tensor = np.load(first_file, allow_pickle=False)
	if first_tensor.shape != (144, 144, 86):
		raise ValueError(f"Expected one raw tensor shaped (144, 144, 86), got {first_tensor.shape} at {first_file}.")
	required_atmospheric_channels = int(atmospheric["num_vertical_levels"]) * int(atmospheric["variables_per_level"])
	if atmospheric["enabled"] and required_atmospheric_channels > first_tensor.shape[2]:
		print(
			"WARNING: atmospheric_features.num_vertical_levels * variables_per_level exceeds the raw channel count. "
			f"Need {required_atmospheric_channels}, raw file has {first_tensor.shape[2]}."
		)
	invalid_low_level_indices = [
		int(index)
		for index in atmospheric["low_level_indices"]
		if int(index) < 0 or int(index) >= int(atmospheric["num_vertical_levels"])
	]
	if atmospheric["enabled"] and invalid_low_level_indices:
		print(
			"WARNING: atmospheric_features.low_level_indices contain invalid z-level indices: "
			f"{invalid_low_level_indices}"
		)

	train_loader, val_loader, test_loader = create_dataloaders(config)
	if len(train_loader.dataset) == 0:
		raise ValueError("Train dataset is empty; cannot run sanity checks.")

	print("Raw data")
	print(f"  path: {first_file}")
	print(f"  shape: {first_tensor.shape}")
	print("Split info")
	print(f"  split_mode: {config.get('split_mode', 'train_val_test')}")
	print(f"  train_fraction: {config.get('train_fraction')}")
	print(f"  val_fraction: {config.get('val_fraction')}")
	print(f"  train samples: {len(train_loader.dataset)}")
	print(f"  val samples: {len(val_loader.dataset)}")
	if test_loader is None:
		print("  external test samples: not configured")
		print("  warning: no external test_data_dir configured; sanity check covers training/validation only")
	else:
		print(f"  external test samples: {len(test_loader.dataset)}")

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

	base_input_channel_count = int(train_loader.dataset.base_input_channel_count)
	fuel_flux_engineered_channel_count = _count_fuel_flux_engineered_channels(config)
	atmospheric_engineered_channel_count = count_atmospheric_engineered_channels(config)
	engineered_channel_slices = resolve_engineered_feature_slices(config, base_input_channel_count)
	configured_model_input_channels = int(config.get("model", {}).get("input_channels", int(x_batch.shape[2])))
	print("Channels")
	print(f"  base input channels: {base_input_channel_count}")
	print(f"  fuel/flux engineered channels: {fuel_flux_engineered_channel_count}")
	print(f"  atmospheric engineered channels: {atmospheric_engineered_channel_count}")
	print(f"  total input channels: {int(x_batch.shape[2])}")
	print(f"  model.input_channels: {configured_model_input_channels}")
	if configured_model_input_channels != int(x_batch.shape[2]):
		print(
			"WARNING: model.input_channels does not match the actual dataset input width. "
			f"Configured={configured_model_input_channels}, actual={int(x_batch.shape[2])}."
		)

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
	if atmospheric_engineered_channel_count > 0:
		if "horizontal_wind_speed" in engineered_channel_slices:
			print(_format_stats("  horizontal wind speed channels", _tensor_stats(x_batch[:, :, engineered_channel_slices["horizontal_wind_speed"], :, :])))
		if "low_level_mean_wind_speed" in engineered_channel_slices:
			print(_format_stats("  low-level mean wind speed channel", _tensor_stats(x_batch[:, :, engineered_channel_slices["low_level_mean_wind_speed"], :, :])))
		if "updraft" in engineered_channel_slices:
			print(_format_stats("  updraft channels", _tensor_stats(x_batch[:, :, engineered_channel_slices["updraft"], :, :])))

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

	if test_loader is not None and len(test_loader.dataset) > 0:
		external_first_file = test_loader.dataset.file_paths[0]
		external_raw = np.load(external_first_file, allow_pickle=False)
		print("External test spatial check")
		print(f"  external raw path: {external_first_file}")
		print(f"  external raw shape: {tuple(external_raw.shape)}")
		external_batch = next(iter(test_loader))
		external_x, external_y = external_batch[:2]
		print(f"  external X before spatial handling: {tuple(external_x.shape)}")
		external_x_device = external_x.to(device)
		external_spatial_result = infer_with_external_test_spatial_handling(model, external_x_device, config)
		external_pred = external_spatial_result["y_pred"]
		external_model_input = external_spatial_result["x_model_input"]
		print(f"  external spatial mode used: {external_spatial_result['mode_used']}")
		if external_spatial_result.get("warning"):
			print(f"  warning: {external_spatial_result['warning']}")
		print(f"  external X fed to model: {tuple(external_model_input.shape)}")
		print(f"  external prediction after crop: {tuple(external_pred.shape)}")
		print(f"  external y shape: {tuple(external_y.shape)}")
		if tuple(external_pred.shape[-2:]) != tuple(external_y.shape[-2:]):
			raise ValueError(
				"External prediction spatial shape does not match external target after crop. "
				f"Prediction={tuple(external_pred.shape)} target={tuple(external_y.shape)}."
			)

	print(f"Finite total loss: {float(total_loss.item()):.6f}")
	print("Sanity check passed")


if __name__ == "__main__":
	main()
