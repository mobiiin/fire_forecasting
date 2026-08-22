"""Sanity-check the wildfire forecasting project before training."""

from __future__ import annotations

import argparse
import json
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
	metadata_batch_to_list,
	_resolve_atmospheric_features_config,
	_resolve_multitask_config,
	create_dataloaders,
	resolve_engineered_feature_slices,
)
from src.data.discovery import discover_multiple_datasets
from src.data.spatial_transforms import infer_with_external_test_spatial_handling
from src.models.model_factory import build_model_from_config
from src.training.input_normalization import apply_input_normalization, build_input_normalizer_for_loader, normalization_metadata_from_loader
from src.training.losses import get_loss_function
from src.training.batch_utils import unpack_batch
from src.training.metrics import compute_metrics
from src.training.model_outputs import extract_prediction
from src.training.train import resolve_validation_policy


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


def _print_metadata_preview(label: str, batch) -> None:
	"""Print a short metadata preview when a batch includes per-sample metadata."""

	if not isinstance(batch, (tuple, list)) or len(batch) < 3:
		return
	metadata_batch = batch[2]
	if not isinstance(metadata_batch, Mapping):
		return
	metadata_items = metadata_batch_to_list(metadata_batch)[:3]
	if not metadata_items:
		return
	print(f"{label} metadata preview")
	for item in metadata_items:
		dataset_name = item.get("dataset_name", "n/a")
		dataset_id = item.get("dataset_id", "n/a")
		sample_index = item.get("sample_index", "n/a")
		current_file = item.get("current_file_path", item.get("current_file", "n/a"))
		target_file = item.get("target_file_path", item.get("future_file", "n/a"))
		print(
			f"  dataset_name={dataset_name} dataset_id={dataset_id} "
			f"sample_index={sample_index} current={current_file} future={target_file}"
		)


def build_arg_parser() -> argparse.ArgumentParser:
	"""Build CLI parser."""

	parser = argparse.ArgumentParser(description="Sanity-check the wildfire forecasting project.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
	parser.add_argument("--batch_size", type=int, default=2, help="Small DataLoader batch size used only for this sanity check.")
	parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers used only for this sanity check.")
	parser.add_argument("--device", default=None, help="Optional device override for this sanity check, e.g. cpu, cuda, cuda:0.")
	parser.add_argument("--deep", action="store_true", help="Run loss/metrics and a backward-pass smoke test.")
	return parser


def _apply_sanity_overrides(config: dict[str, Any], batch_size: int, num_workers: int, device: str | None) -> None:
	"""Keep the sanity check lightweight without mutating the saved config."""

	sanity_batch_size = max(1, int(batch_size))
	sanity_num_workers = max(0, int(num_workers))
	config["batch_size"] = sanity_batch_size
	training_config = config.get("training")
	if not isinstance(training_config, dict):
		training_config = {}
	training_config["batch_size"] = sanity_batch_size
	training_config["num_workers"] = sanity_num_workers
	if device not in (None, "", "null"):
		training_config["device"] = str(device)
	config["training"] = training_config
	data_loader_config = config.get("data_loader")
	if not isinstance(data_loader_config, dict):
		data_loader_config = {}
	data_loader_config["batch_size"] = sanity_batch_size
	data_loader_config["num_workers"] = sanity_num_workers
	for split in ("train", "val", "test"):
		split_config = data_loader_config.get(split)
		if not isinstance(split_config, dict):
			split_config = {}
		split_config["batch_size"] = sanity_batch_size
		split_config["num_workers"] = sanity_num_workers
		split_config["persistent_workers"] = False
		data_loader_config[split] = split_config
	config["data_loader"] = data_loader_config


def main() -> None:
	"""Run project sanity checks end-to-end."""

	args = build_arg_parser().parse_args()
	if torch is None:
		raise ImportError("PyTorch is required for sanity_check_project.py.")
	config_path = Path(args.config).expanduser().resolve()
	config = load_config(config_path)
	config["config_path"] = str(config_path)
	_apply_sanity_overrides(config, args.batch_size, args.num_workers, args.device)

	if str(config.get("dataloader", {}).get("source", "")).lower() == "processed_full_frames":
		processed = config.get("processed_dataset", {}) if isinstance(config.get("processed_dataset"), Mapping) else {}
		root = Path(str(processed.get("root", ""))).expanduser()
		pattern = str(config.get("dataloader", {}).get("sample_pattern", "consecutive5_h10"))
		checks = [root, root / "dataset_manifest.json", root / "channel_manifest.json", root / "indices" / "temporal" / f"samples_{pattern}.jsonl"]
		for path in checks:
			if not path.exists(): raise FileNotFoundError(f"Processed dataset sanity check failed; missing: {path}")
		manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
		channel_manifest = json.loads((root / "channel_manifest.json").read_text(encoding="utf-8"))
		if int(channel_manifest.get("num_channels", channel_manifest.get("channels", 0) if isinstance(channel_manifest.get("channels"), int) else len(channel_manifest.get("channels", [])))) <= 0:
			print("WARNING: channel_manifest did not expose a positive channel count; continuing with DataLoader inference.")
		manifest_split_fires = {}
		manifest_splits = manifest.get("splits", {}) if isinstance(manifest.get("splits"), Mapping) else {}
		for split in ("train", "val", "test"):
			fires = manifest_splits.get(f"{split}_fires", manifest_splits.get(split, []))
			manifest_split_fires[split] = {str(fire) for fire in fires} if isinstance(fires, list) else set()
			if not manifest_split_fires[split]:
				print(f"WARNING: dataset_manifest has no {split} fire list; DataLoader split records will be authoritative.")
		train_loader, val_loader, test_loader = create_dataloaders(config)
		if not len(train_loader.dataset) or not len(val_loader.dataset) or not len(test_loader.dataset): raise ValueError("Processed dataset sanity check requires non-empty train/val/test splits.")
		loader_fire_sets = {split: {str(record.get("fire_name")) for record in loader.dataset.records} for split, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader))}
		for split, manifest_fires in manifest_split_fires.items():
			if manifest_fires and loader_fire_sets[split] != manifest_fires:
				raise ValueError(f"Processed {split} fires in sample index do not match dataset_manifest: index={sorted(loader_fire_sets[split])} manifest={sorted(manifest_fires)}")
		for split, fire_set in loader_fire_sets.items():
			for fire in sorted(fire_set):
				terrain_path = root / "fires" / fire / "terrain" / "terrain_features.npy"
				if bool(config.get("cawfe_latte", {}).get("use_terrain_conditioning", False)) and not terrain_path.exists():
					raise FileNotFoundError(f"Missing terrain features for {split} fire {fire}: {terrain_path}")
		batch = next(iter(train_loader)); x_batch, y_batch, batch_extra = unpack_batch(batch); terrain_batch = batch_extra.get("terrain")
		if x_batch.ndim != 5 or y_batch.ndim != 4 or y_batch.shape[1] != 4: raise ValueError(f"Processed batch shapes must be X=(B,T,C,H,W), y=(B,4,H,W); got {tuple(x_batch.shape)}, {tuple(y_batch.shape)}")
		if not torch.isfinite(x_batch).all(): raise ValueError("Processed inputs contain NaN/Inf values.")
		expected_t = int(config.get("input_sequence_length", config.get("training", {}).get("input_sequence_length", x_batch.shape[1])))
		if int(x_batch.shape[1]) != expected_t: raise ValueError(f"Processed input T={x_batch.shape[1]} does not match configured T={expected_t}")
		mask_values = torch.unique(y_batch[:, 2])
		if not bool(torch.all((mask_values == 0) | (mask_values == 1))): raise ValueError(f"Processed fire mask is not binary: {mask_values.tolist()}")
		if not torch.isfinite(y_batch[:, 0]).all() or not torch.isfinite(y_batch[:, 1]).all() or not torch.isfinite(y_batch[:, 3]).all(): raise ValueError("Processed targets contain NaN/Inf values.")
		if bool((y_batch[:, 3] < 0).any()): raise ValueError("Processed energy_log targets must be non-negative.")
		if getattr(train_loader.dataset, "input_normalization_on_device", False): raise ValueError("Processed dataset unexpectedly enables device-side normalization.")
		if not getattr(train_loader.dataset, "inputs_are_normalized", False) and bool(config.get("dataloader", {}).get("normalize_inputs", True)): raise ValueError("Processed inputs were not normalized in the Dataset.")
		model = build_model_from_config(config, input_channels=int(x_batch.shape[2]))
		if bool(config.get("cawfe_latte", {}).get("use_terrain_conditioning", False)) and terrain_batch is None: raise ValueError("CAWFE-Latte terrain conditioning is enabled but sanity batch has no terrain.")
		if terrain_batch is not None:
			if terrain_batch.ndim != 4 or int(terrain_batch.shape[1]) != 4 or tuple(terrain_batch.shape[-2:]) != tuple(y_batch.shape[-2:]): raise ValueError(f"Terrain batch must be B,4,H,W aligned with targets; got terrain={tuple(terrain_batch.shape)} y={tuple(y_batch.shape)}")
			if not torch.isfinite(terrain_batch).all(): raise ValueError("Terrain batch contains NaN/Inf values.")
			ranges = [(0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
			for channel, (lo, hi) in enumerate(ranges):
				ch = terrain_batch[:, channel]
				if float(ch.min()) < lo - 1.0e-4 or float(ch.max()) > hi + 1.0e-4: raise ValueError(f"Terrain channel {channel} outside expected range [{lo}, {hi}]: min={float(ch.min())} max={float(ch.max())}")
		with torch.no_grad(): model_output = model(x_batch, terrain=terrain_batch) if terrain_batch is not None else model(x_batch)
		prediction = extract_prediction(model_output)
		if tuple(prediction.shape) != (int(x_batch.shape[0]), 4, int(y_batch.shape[2]), int(y_batch.shape[3])): raise ValueError(f"Processed model output shape mismatch: {tuple(prediction.shape)}")
		criterion = get_loss_function(config)
		loss_result = criterion(model_output, y_batch)
		loss_value = loss_result["total_loss"] if isinstance(loss_result, Mapping) else loss_result
		if not torch.isfinite(loss_value): raise ValueError("Processed loss is not finite.")
		metric_values = compute_metrics(prediction, y_batch, config)
		if args.deep:
			model.zero_grad(set_to_none=True); deep_output = model(x_batch, terrain=terrain_batch) if terrain_batch is not None else model(x_batch); deep_loss_result = criterion(deep_output, y_batch); deep_loss = deep_loss_result["total_loss"] if isinstance(deep_loss_result, Mapping) else deep_loss_result; deep_loss.backward()
		print("Data pipeline mode: processed_full_frames")
		print(f"  root: {root}")
		print(f"  pattern: {pattern}")
		print(f"  train samples: {len(train_loader.dataset)}")
		print(f"  val samples: {len(val_loader.dataset)}")
		print(f"  test samples: {len(test_loader.dataset)}")
		print(f"  X batch shape: {tuple(x_batch.shape)}")
		print(f"  y batch shape: {tuple(y_batch.shape)}")
		print(f"  input normalization applied in dataset: {getattr(train_loader.dataset, 'inputs_are_normalized', False)}")
		print(f"  device-side normalization skipped: {not getattr(train_loader.dataset, 'input_normalization_on_device', False)}")
		print(f"  loss: {float(loss_value.item()):.6g}")
		print(f"  metrics: {metric_values}")
		print(f"  deep backward: {args.deep}")
		print("Status: OK")
		return

	_print_environment_info()
	data_dir_value = config.get("data_dir")
	if data_dir_value in (None, "", "null"):
		dataset_records = discover_multiple_datasets(config)
		if not dataset_records:
			raise ValueError("No dataset records were discovered for sanity checks.")
		raw_files = list(dataset_records[0]["file_paths"])
		data_dir = Path(dataset_records[0]["data_dir"])
	else:
		data_dir = _resolve_path(config_path, data_dir_value)
		raw_files = _sort_chronologically(list(data_dir.glob(str(config["file_pattern"]))))
	atmospheric = _resolve_atmospheric_features_config(config)
	if not raw_files:
		raise FileNotFoundError(f"No files found in '{data_dir}' using pattern '{config['file_pattern']}'.")
	first_file = raw_files[0]
	first_tensor = np.load(first_file, allow_pickle=False)
	if first_tensor.ndim != 3:
		raise ValueError(f"Expected one raw tensor shaped (H, W, C), got {first_tensor.shape} at {first_file}.")
	expected_raw_channels = int(config.get("input_channel_count", config.get("model", {}).get("raw_input_channels", first_tensor.shape[2])))
	if int(first_tensor.shape[2]) != expected_raw_channels:
		raise ValueError(
			"Raw tensor channel count does not match the configured base input width. "
			f"Expected C={expected_raw_channels}, got shape={first_tensor.shape} at {first_file}."
		)
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
	if atmospheric["enabled"] and atmospheric["add_wind_direction"] and atmospheric["wind_direction_mode"] != "unit_vector":
		print(
			"WARNING: atmospheric_features.wind_direction_mode is not 'unit_vector': "
			f"{atmospheric['wind_direction_mode']!r}"
		)
	if atmospheric["enabled"] and atmospheric["add_wind_direction"] and atmospheric["wind_direction_convention"] != "toward":
		print(
			"WARNING: atmospheric_features.wind_direction_convention is not 'toward': "
			f"{atmospheric['wind_direction_convention']!r}"
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

	training_config = config.get("training", {}) if isinstance(config.get("training"), Mapping) else {}
	performance_config = training_config.get("performance", {}) if isinstance(training_config.get("performance"), Mapping) else {}
	validation_policy = resolve_validation_policy(config, val_loader=val_loader)
	print("Validation")
	print(f"  mode: {validation_policy['validation_mode']}")
	print(f"  max_val_batches_per_epoch: {validation_policy['max_val_batches_per_epoch']}")
	print(f"  fixed_subset_seed: {validation_policy['fixed_subset_seed']}")
	print("  checkpoint selection: same validation protocol")
	print("  status: OK")
	if "full_validation_every_n_epochs" in config or "full_validation_every_n_epochs" in training_config or "full_validation_every_n_epochs" in performance_config:
		print("  WARNING: full_validation_every_n_epochs is deprecated and ignored. Use training.validation.mode instead.")
	if validation_policy["validation_mode"] == "full_every_epoch":
		validation_config = training_config.get("validation", {}) if isinstance(training_config.get("validation"), Mapping) else {}
		if validation_config.get("max_val_batches_per_epoch") not in (None, "", "null", "None"):
			print("  WARNING: max_val_batches_per_epoch is ignored when mode is full_every_epoch.")
	early_config = training_config.get("early_stopping", {}) if isinstance(training_config.get("early_stopping"), Mapping) else {}
	checkpointing_config = training_config.get("checkpointing", {}) if isinstance(training_config.get("checkpointing"), Mapping) else {}
	early_monitor = str(early_config.get("monitor", "val_loss"))
	early_mode = str(early_config.get("mode", "min")).lower()
	print("Early stopping")
	print(f"  enabled: {bool(early_config.get('enabled', False))}")
	print(f"  monitor: {early_monitor}")
	print(f"  mode: {early_mode}")
	print(f"  patience: {early_config.get('patience', 8)}")
	print(f"  min_delta: {early_config.get('min_delta', 0.001)}")
	print(f"  start_epoch: {early_config.get('start_epoch', 5)}")
	print(f"  stop_on_nan: {bool(early_config.get('stop_on_nan', True))}")
	print("  status: OK")
	if early_mode not in {"min", "max"}:
		print("  WARNING: early_stopping.mode should be 'min' or 'max'.")
	checkpoint_monitor = str(checkpointing_config.get("monitor", "val_loss"))
	if early_monitor != checkpoint_monitor:
		print(f"  WARNING: early stopping monitor {early_monitor!r} differs from checkpoint monitor {checkpoint_monitor!r}.")

	train_batch = next(iter(train_loader))
	x_batch, y_batch = train_batch[:2]
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
	input_normalizer = build_input_normalizer_for_loader(train_loader, device, int(x_batch.shape[2]), config)
	x_batch = apply_input_normalization(x_batch, input_normalizer, config)

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

	expected_y_channels = int(config.get("model", {}).get("output_channels", 3 if task_type == "multitask" else 1))
	if y_batch.shape[1] != expected_y_channels:
		raise ValueError(f"Expected y channel dimension {expected_y_channels}, got {y_batch.shape[1]}")
	if tuple(y_pred.shape) != tuple(y_batch.shape):
		raise ValueError(f"Expected model output shape {tuple(y_batch.shape)}, got {tuple(y_pred.shape)}")

	print("Shapes")
	print(f"  X batch shape: {tuple(x_batch.shape)}")
	print(f"  y batch shape: {tuple(y_batch.shape)}")
	print(f"  model output shape: {tuple(y_pred.shape)}")
	print("Input normalization")
	print(f"  train: {normalization_metadata_from_loader(train_loader, config, int(x_batch.shape[2]))}")
	print(f"  val: {normalization_metadata_from_loader(val_loader, config, int(x_batch.shape[2]))}")
	if test_loader is not None:
		print(f"  test: {normalization_metadata_from_loader(test_loader, config, int(x_batch.shape[2]))}")
	if str(config.get("split_mode", "")).lower() == "multi_dataset_chronological":
		train_metadata_items = metadata_batch_to_list(train_batch[2]) if len(train_batch) >= 3 and isinstance(train_batch[2], Mapping) else []
		for item in train_metadata_items[:3]:
			data_dir = str(item.get("data_dir", ""))
			current_file = str(item.get("current_file_path", item.get("current_file", "")))
			target_file = str(item.get("target_file_path", item.get("future_file", "")))
			if data_dir and not current_file.startswith(data_dir):
				raise ValueError(
					"Multi-dataset metadata sanity check failed: current_file_path is outside the sample's data_dir. "
					f"data_dir={data_dir} current_file={current_file}"
				)
			if data_dir and not target_file.startswith(data_dir):
				raise ValueError(
					"Multi-dataset metadata sanity check failed: target_file_path is outside the sample's data_dir. "
					f"data_dir={data_dir} target_file={target_file}"
				)
		if not hasattr(train_loader.dataset, "initial_fuel_maps") or not getattr(train_loader.dataset, "initial_fuel_maps"):
			raise ValueError("Multi-dataset sanity check failed: dataset.initial_fuel_maps is missing or empty.")
		_print_metadata_preview("train", train_batch)
		_print_metadata_preview("val", next(iter(val_loader)))
		if test_loader is not None and len(test_loader.dataset) > 0:
			_print_metadata_preview("test", next(iter(test_loader)))
	if atmospheric_engineered_channel_count > 0:
		if "horizontal_wind_speed" in engineered_channel_slices:
			print(_format_stats("  horizontal wind speed channels", _tensor_stats(x_batch[:, :, engineered_channel_slices["horizontal_wind_speed"], :, :])))
		if "low_level_mean_wind_speed" in engineered_channel_slices:
			print(_format_stats("  low-level mean wind speed channel", _tensor_stats(x_batch[:, :, engineered_channel_slices["low_level_mean_wind_speed"], :, :])))
		if "updraft" in engineered_channel_slices:
			print(_format_stats("  updraft channels", _tensor_stats(x_batch[:, :, engineered_channel_slices["updraft"], :, :])))
		if "wind_dir_cos" in engineered_channel_slices:
			print(_format_stats("  wind_dir_cos channels", _tensor_stats(x_batch[:, :, engineered_channel_slices["wind_dir_cos"], :, :])))
		if "wind_dir_sin" in engineered_channel_slices:
			print(_format_stats("  wind_dir_sin channels", _tensor_stats(x_batch[:, :, engineered_channel_slices["wind_dir_sin"], :, :])))

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
		print("External test spatial check")
		external_file_paths = getattr(test_loader.dataset, "file_paths", None)
		if external_file_paths:
			external_first_file = external_file_paths[0]
			external_raw = np.load(external_first_file, allow_pickle=False)
			print(f"  external raw path: {external_first_file}")
			print(f"  external raw shape: {tuple(external_raw.shape)}")
		else:
			cache_dir = getattr(test_loader.dataset, "cache_dir", None)
			shards = getattr(test_loader.dataset, "shards", [])
			first_shard = shards[0].get("path") if shards and isinstance(shards[0], Mapping) else None
			print(f"  external raw path: not available from {type(test_loader.dataset).__name__}")
			if cache_dir is not None:
				print(f"  cache_dir: {cache_dir}")
			if first_shard is not None:
				print(f"  first cache shard: {first_shard}")
			metadata = getattr(test_loader.dataset, "metadata", [])
			if metadata:
				first_metadata = metadata[0]
				current_file = first_metadata.get("current_file_path", first_metadata.get("current_file"))
				target_file = first_metadata.get("target_file_path", first_metadata.get("future_file"))
				if current_file:
					print(f"  metadata current file: {current_file}")
				if target_file:
					print(f"  metadata target file: {target_file}")
		external_batch = next(iter(test_loader))
		external_x, external_y = external_batch[:2]
		print(f"  external X before spatial handling: {tuple(external_x.shape)}")
		external_x_device = external_x.to(device)
		external_normalizer = build_input_normalizer_for_loader(test_loader, device, int(external_x_device.shape[2]), config)
		external_x_device = apply_input_normalization(external_x_device, external_normalizer, config)
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
