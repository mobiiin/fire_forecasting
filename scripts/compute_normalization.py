"""Compute channel-wise normalization statistics from training samples."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.config import load_config
from src.data.discovery import discover_dataset_files, discover_multiple_datasets
from src.data.dataset import (
	FireSequenceDataset,
	MultiFirePatchSequenceDataset,
	MultiFireSequenceDataset,
	_count_fuel_flux_engineered_channels,
	count_atmospheric_engineered_channels,
)
from src.data.patching import resolve_patching_config
from src.data.splits import (
	manual_fire_holdout_splits,
	chronological_split_indices,
	chronological_train_val_split_indices,
	chronological_split_indices_for_dataset,
	multi_dataset_chronological_splits,
	multi_fire_chronological_splits,
)


def _resolve_path(base_path: Path, configured_path: str | Path) -> Path:
	"""Resolve a configured path relative to the config file location."""

	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	return (base_path.parent / path).resolve()


def _update_running_stats(
	array: np.ndarray,
	count: int,
	mean: np.ndarray | None,
	m2: np.ndarray | None,
	channel_min: np.ndarray | None,
	channel_max: np.ndarray | None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Update running per-channel stats from an array shaped (..., C)."""

	flat = np.asarray(array, dtype=np.float64).reshape(-1, array.shape[-1])
	file_count = flat.shape[0]
	file_mean = flat.mean(axis=0)
	file_min = flat.min(axis=0)
	file_max = flat.max(axis=0)
	centered = flat - file_mean
	file_m2 = np.sum(centered * centered, axis=0)

	if mean is None or m2 is None or channel_min is None or channel_max is None:
		return file_count, file_mean, file_m2, file_min, file_max

	delta = file_mean - mean
	total_count = count + file_count
	mean = mean + delta * (file_count / total_count)
	m2 = m2 + file_m2 + (delta * delta) * (count * file_count / total_count)
	channel_min = np.minimum(channel_min, file_min)
	channel_max = np.maximum(channel_max, file_max)
	return total_count, mean, m2, channel_min, channel_max


def build_arg_parser() -> argparse.ArgumentParser:
	"""Create the command-line interface for normalization computation."""

	parser = argparse.ArgumentParser(description="Compute normalization statistics.")
	parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument("--from_cache", action="store_true", help="Compute stats from the precomputed train patch cache.")
	return parser


def _save_stats(
	config_path: Path,
	normalization_config: Mapping[str, Any],
	stats: dict[str, np.ndarray],
) -> Path:
	"""Save normalization stats to the configured archive path."""

	output_path = _resolve_path(config_path, normalization_config["path"])
	output_path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(output_path, **stats)
	return output_path


def main() -> None:
	"""Compute and save normalization statistics for the training split only."""

	args = build_arg_parser().parse_args()
	config_path = Path(args.config).expanduser().resolve()
	config = load_config(config_path)
	config["config_path"] = str(config_path)

	for required_key in ("data_dir", "file_pattern", "input_sequence_length", "prediction_horizon", "train_fraction", "val_fraction"):
		if required_key not in config:
			raise KeyError(f"Config is missing required key '{required_key}'.")

	normalization_config = dict(config.get("normalization", {}))
	if "path" not in normalization_config:
		raise KeyError("Config is missing normalization.path.")

	compute_from_cache = bool(args.from_cache or normalization_config.get("compute_from_patch_cache", False))
	if compute_from_cache:
		from src.data.cache import get_patch_cache_dir, validate_patch_cache
		from src.data.cached_patch_dataset import CachedPatchDataset

		cache_config = dict(config.get("cache", {})) if isinstance(config.get("cache"), dict) else {}
		if bool(cache_config.get("save_normalized_inputs", False)):
			raise ValueError(
				"Cannot compute raw input normalization from cache because cache.save_normalized_inputs=true. "
				"Use an unnormalized train patch cache or keep the existing stats file."
			)
		validate_patch_cache(config, split="train")
		train_dataset = CachedPatchDataset(
			cache_dir=get_patch_cache_dir(config),
			split="train",
			config=config,
			normalization_stats=None,
			return_metadata=False,
		)
		first_x_tensor, _ = train_dataset[0][:2]
		actual_input_channel_count = int(first_x_tensor.shape[1])
		configured_model_input_channels = int(config.get("model", {}).get("input_channels", actual_input_channel_count))
		if configured_model_input_channels != actual_input_channel_count:
			raise ValueError(
				"model.input_channels does not match the cached train patch input width. "
				f"Configured model.input_channels={configured_model_input_channels}, cached channels={actual_input_channel_count}."
			)

		count = 0
		mean = None
		m2 = None
		channel_min = None
		channel_max = None
		for sample_index in range(len(train_dataset)):
			x_tensor, _ = train_dataset[sample_index][:2]
			x_array = x_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
			count, mean, m2, channel_min, channel_max = _update_running_stats(
				x_array,
				count,
				mean,
				m2,
				channel_min,
				channel_max,
			)
			if (sample_index + 1) % 500 == 0:
				print(f"cache normalization: processed {sample_index + 1}/{len(train_dataset)} train patches")

		if mean is None or m2 is None or channel_min is None or channel_max is None:
			raise ValueError("Failed to compute any input normalization statistics from the patch cache.")
		eps = float(normalization_config.get("epsilon", 1e-6))
		variance = m2 / max(count, 1)
		std = np.sqrt(np.maximum(variance, 0.0))
		std = np.maximum(std, eps)
		stats: dict[str, np.ndarray] = {
			"mean": mean.astype(np.float32),
			"std": std.astype(np.float32),
			"min": channel_min.astype(np.float32),
			"max": channel_max.astype(np.float32),
			"input_channel_count": np.asarray(train_dataset.total_input_channels, dtype=np.int64),
			"base_input_channel_count": np.asarray(train_dataset.base_input_channel_count, dtype=np.int64),
			"fuel_flux_engineered_channel_count": np.asarray(train_dataset.fuel_flux_engineered_channel_count, dtype=np.int64),
			"atmospheric_engineered_channel_count": np.asarray(train_dataset.atmospheric_engineered_channel_count, dtype=np.int64),
			"engineered_channel_count": np.asarray(train_dataset.engineered_channel_count, dtype=np.int64),
		}
		output_path = _save_stats(config_path, normalization_config, stats)
		channel_mean = stats["mean"]
		channel_std = stats["std"]
		near_zero_std = int(np.sum(channel_std <= max(eps, 1e-6) * 10.0))
		print(f"C: {channel_mean.shape[0]}")
		print("split mode: patch_cache")
		print(f"train patches used for normalization: {len(train_dataset)}")
		print(f"raw/base input channels: {train_dataset.base_input_channel_count}")
		print(f"fuel/flux engineered channels: {train_dataset.fuel_flux_engineered_channel_count}")
		print(f"atmospheric engineered channels: {train_dataset.atmospheric_engineered_channel_count}")
		print(f"total model input channels: {train_dataset.total_input_channels}")
		print(f"saved stats shape: mean={channel_mean.shape} std={channel_std.shape}")
		print(f"global channel mean range: {channel_mean.min():.6g} to {channel_mean.max():.6g}")
		print(f"global channel std range: {channel_std.min():.6g} to {channel_std.max():.6g}")
		print(f"channels with near-zero std: {near_zero_std}")
		print(f"output path: {output_path}")
		return

	split_mode = str(config.get("split_mode", "train_val_test")).lower()
	if split_mode == "multi_dataset_chronological":
		split_mode = "multi_fire_chronological"
	train_fraction = float(config["train_fraction"])
	val_fraction = float(config["val_fraction"])
	test_fraction = float(config.get("test_fraction", 0.0))
	file_pattern = str(config["file_pattern"])
	patching_config = resolve_patching_config(config)
	patch_size = int(patching_config["patch_height"])
	if patch_size != int(patching_config["patch_width"]):
		raise ValueError(
			"compute_normalization currently requires square patches. "
			f"Got patch_height={patching_config['patch_height']} patch_width={patching_config['patch_width']}."
		)
	if split_mode in {"multi_fire_chronological", "manual_fire_holdout"}:
		dataset_records = discover_multiple_datasets(config)
		if split_mode == "manual_fire_holdout":
			manual_section = config.get("manual_fire_split", {}) if isinstance(config.get("manual_fire_split"), dict) else {}
			sample_refs = manual_fire_holdout_splits(
				dataset_records=dataset_records,
				train_fire_names=manual_section.get("train_fires", []),
				val_fire_names=manual_section.get("val_fires", []),
				test_fire_names=manual_section.get("test_fires", []),
				input_sequence_length=int(config["input_sequence_length"]),
				prediction_horizon=int(config["prediction_horizon"]),
				config=config,
			)
		else:
			sample_refs = multi_fire_chronological_splits(
				dataset_records=dataset_records,
				input_sequence_length=int(config["input_sequence_length"]),
				prediction_horizon=int(config["prediction_horizon"]),
				train_fraction=train_fraction,
				val_fraction=val_fraction,
				test_fraction=test_fraction,
			)
		train_dataset = MultiFirePatchSequenceDataset(
			dataset_records=dataset_records,
			sample_refs=sample_refs["train"],
			input_sequence_length=int(config["input_sequence_length"]),
			prediction_horizon=int(config["prediction_horizon"]),
			target_channel=int(config.get("target_channel", 0)),
			input_channel_count=int(config.get("input_channel_count", config.get("model", {}).get("input_channels", 0))),
			input_channel_indices=config.get("input_channel_indices"),
			task_type=str(config.get("task_type", config.get("training", {}).get("task_type", "regression"))),
			fire_threshold=float(config.get("fire_threshold", config.get("training", {}).get("fire_threshold", 0.5))),
			use_patches=bool(patching_config["enabled"]),
			patch_size=patch_size,
			active_patch_probability=float(patching_config["active_patch_probability"]),
			active_threshold=float(config.get("active_threshold", config.get("fire_threshold", 0.5))),
			normalization_stats=None,
			normalize_target=False,
			config=config,
			split="train",
		)
		per_dataset_train_counts: dict[str, int] = {}
		per_dataset_val_counts: dict[str, int] = {}
		per_dataset_test_counts: dict[str, int] = {}
		for dataset_record in dataset_records:
			dataset_name = str(dataset_record["dataset_name"])
			per_dataset_train_counts[dataset_name] = sum(1 for ref in sample_refs["train"] if str(ref["dataset_name"]) == dataset_name)
			per_dataset_val_counts[dataset_name] = sum(1 for ref in sample_refs["val"] if str(ref["dataset_name"]) == dataset_name)
			per_dataset_test_counts[dataset_name] = sum(1 for ref in sample_refs["test"] if str(ref["dataset_name"]) == dataset_name)
		splits = {
			"train": sample_refs["train"],
			"val": sample_refs["val"],
			"test": sample_refs["test"],
		}
	else:
		data_dir = _resolve_path(config_path, config["data_dir"])
		if not data_dir.exists():
			raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
		files = discover_dataset_files(data_dir, file_pattern)
		if split_mode == "train_val_external_test":
			splits = {
				**chronological_train_val_split_indices(
					num_timesteps=len(files),
					input_sequence_length=int(config["input_sequence_length"]),
					prediction_horizon=int(config["prediction_horizon"]),
					train_fraction=train_fraction,
					val_fraction=val_fraction,
				),
				"test": [],
			}
		else:
			splits = chronological_split_indices(
				num_timesteps=len(files),
				input_sequence_length=int(config["input_sequence_length"]),
				prediction_horizon=int(config["prediction_horizon"]),
				train_fraction=train_fraction,
				val_fraction=val_fraction,
				test_fraction=test_fraction,
				split_mode=split_mode,
			)
		if not splits["train"]:
			raise ValueError("No training samples were found for normalization.")

		train_dataset = FireSequenceDataset(
			file_paths=files,
			sample_indices=splits["train"],
			input_sequence_length=int(config["input_sequence_length"]),
			prediction_horizon=int(config["prediction_horizon"]),
			target_channel=int(config.get("target_channel", 0)),
			input_channel_count=int(config.get("input_channel_count", config.get("model", {}).get("input_channels", 0))),
			input_channel_indices=config.get("input_channel_indices"),
			task_type=str(config.get("task_type", config.get("training", {}).get("task_type", "regression"))),
			fire_threshold=float(config.get("fire_threshold", config.get("training", {}).get("fire_threshold", 0.5))),
			use_patches=False,
			patch_size=patch_size,
			active_patch_probability=float(patching_config["active_patch_probability"]),
			active_threshold=float(config.get("active_threshold", config.get("fire_threshold", 0.5))),
			normalization_stats=None,
			normalize_target=False,
			config=config,
		)
		per_dataset_train_counts = {}
		per_dataset_val_counts = {}
		per_dataset_test_counts = {}

	if not splits["train"]:
		raise ValueError("No training samples were found for normalization.")
	first_x_tensor, _ = train_dataset[0][:2]
	actual_input_channel_count = int(first_x_tensor.shape[1])
	configured_model_input_channels = int(config.get("model", {}).get("input_channels", actual_input_channel_count))
	if actual_input_channel_count != int(train_dataset.total_input_channels):
		raise ValueError(
			"Dataset-reported total_input_channels does not match actual sample tensor shape. "
			f"Dataset reports {train_dataset.total_input_channels}, sample has {actual_input_channel_count}."
		)
	if configured_model_input_channels != actual_input_channel_count:
		raise ValueError(
			"model.input_channels does not match the actual engineered dataset input width. "
			f"Configured model.input_channels={configured_model_input_channels}, actual dataset channels={actual_input_channel_count}."
		)

	count = 0
	mean = None
	m2 = None
	channel_min = None
	channel_max = None
	target_count = 0
	target_mean = None
	target_m2 = None
	target_min = None
	target_max = None
	task_type = str(config.get("task_type", "regression")).lower()
	target_normalization_config = dict(config.get("target_normalization", {}))
	normalize_targets = bool(target_normalization_config.get("enabled", False))

	for sample_index in range(len(train_dataset)):
		x_tensor, y_tensor = train_dataset[sample_index][:2]
		x_array = x_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
		count, mean, m2, channel_min, channel_max = _update_running_stats(x_array, count, mean, m2, channel_min, channel_max)

		if not normalize_targets:
			continue
		y_array = y_tensor.detach().cpu().numpy().transpose(1, 2, 0)
		if task_type == "regression":
			y_array = y_array[:, :, :1]
		elif task_type == "multitask":
			y_array = y_array[:, :, :2]
		else:
			continue
		target_count, target_mean, target_m2, target_min, target_max = _update_running_stats(
			y_array,
			target_count,
			target_mean,
			target_m2,
			target_min,
			target_max,
		)

	if mean is None or m2 is None or channel_min is None or channel_max is None:
		raise ValueError("Failed to compute any input normalization statistics.")

	eps = float(normalization_config.get("epsilon", 1e-6))
	variance = m2 / max(count, 1)
	std = np.sqrt(np.maximum(variance, 0.0))
	std = np.maximum(std, eps)
	stats: dict[str, np.ndarray] = {
		"mean": mean.astype(np.float32),
		"std": std.astype(np.float32),
		"min": channel_min.astype(np.float32),
		"max": channel_max.astype(np.float32),
		"input_channel_count": np.asarray(train_dataset.total_input_channels, dtype=np.int64),
		"base_input_channel_count": np.asarray(train_dataset.base_input_channel_count, dtype=np.int64),
		"fuel_flux_engineered_channel_count": np.asarray(train_dataset.fuel_flux_engineered_channel_count, dtype=np.int64),
		"atmospheric_engineered_channel_count": np.asarray(train_dataset.atmospheric_engineered_channel_count, dtype=np.int64),
		"engineered_channel_count": np.asarray(train_dataset.engineered_channel_count, dtype=np.int64),
	}

	if normalize_targets and target_mean is not None and target_m2 is not None and target_min is not None and target_max is not None:
		target_variance = target_m2 / max(target_count, 1)
		target_std = np.sqrt(np.maximum(target_variance, 0.0))
		target_std = np.maximum(target_std, eps)
		if task_type == "regression":
			stats["target_mean"] = np.asarray(target_mean[0], dtype=np.float32)
			stats["target_std"] = np.asarray(target_std[0], dtype=np.float32)
			stats["target_min"] = np.asarray(target_min[0], dtype=np.float32)
			stats["target_max"] = np.asarray(target_max[0], dtype=np.float32)
		elif task_type == "multitask":
			stats["multitask_target_mean"] = target_mean.astype(np.float32)
			stats["multitask_target_std"] = target_std.astype(np.float32)

	output_path = _resolve_path(config_path, normalization_config["path"])
	output_path.parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(output_path, **stats)

	channel_mean = stats["mean"]
	channel_std = stats["std"]
	near_zero_std = int(np.sum(channel_std <= max(eps, 1e-6) * 10.0))
	if channel_mean.shape[0] != actual_input_channel_count or channel_std.shape[0] != actual_input_channel_count:
		raise ValueError(
			"Saved normalization stats length does not match actual dataset input channel count. "
			f"Expected {actual_input_channel_count}, got mean={channel_mean.shape[0]} std={channel_std.shape[0]}."
		)
	fuel_flux_engineered_channel_count = _count_fuel_flux_engineered_channels(config)
	atmospheric_engineered_channel_count = count_atmospheric_engineered_channels(config)
	print(f"C: {channel_mean.shape[0]}")
	print(f"split mode: {split_mode}")
	if split_mode in {"multi_fire_chronological", "manual_fire_holdout"}:
		print(f"number of datasets: {len(dataset_records)}")
		for dataset_name, train_count in per_dataset_train_counts.items():
			print(
				f"dataset {dataset_name}: normalization_train={train_count} "
				f"val={per_dataset_val_counts.get(dataset_name, 0)} "
				f"test={per_dataset_test_counts.get(dataset_name, 0)}"
			)
		print(f"total train samples used for normalization: {len(splits['train'])}")
		print(f"total val/test samples ignored: {len(splits['val']) + len(splits['test'])}")
	else:
		print(f"train samples used for normalization: {len(splits['train'])}")
		print(f"validation samples not used: {len(splits['val'])}")
		print("external test dataset ignored for normalization")
	print(f"raw/base input channels: {train_dataset.base_input_channel_count}")
	print(f"fuel/flux engineered channels: {fuel_flux_engineered_channel_count}")
	print(f"atmospheric engineered channels: {atmospheric_engineered_channel_count}")
	print(f"total model input channels: {train_dataset.total_input_channels}")
	print(f"saved stats shape: mean={channel_mean.shape} std={channel_std.shape}")
	print(f"global channel mean range: {channel_mean.min():.6g} to {channel_mean.max():.6g}")
	print(f"global channel std range: {channel_std.min():.6g} to {channel_std.max():.6g}")
	print(f"channels with near-zero std: {near_zero_std}")
	print(f"output path: {output_path}")


if __name__ == "__main__":
	main()
