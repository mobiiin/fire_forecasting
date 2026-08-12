"""Compute channel-wise normalization statistics from training samples."""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.config import compute_file_sha256, compute_text_sha256, load_config
from src.data.cache import (
	compute_dataset_index_hash,
	get_patch_cache_dir,
	load_cache_manifest,
	resolve_dataset_index_path,
)
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
from src.training.input_normalization import normalization_config as resolve_normalization_config


NORMALIZATION_VERSION = "v2_timestamped_config_aware"


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
	parser.add_argument("--output_dir", default=None, help="Override normalization output directory.")
	parser.add_argument("--config_name", default=None, help="Override config name used in timestamped filenames.")
	parser.add_argument("--no_latest_alias", action="store_true", help="Do not update latest_train_normalization_stats aliases.")
	parser.add_argument("--latest_as_copy", action="store_true", help="Copy latest aliases instead of symlinking them.")
	return parser


def _sanitize_config_name(value: Any) -> str:
	"""Return a filesystem-safe config name for normalization filenames."""

	sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip()).strip("_")
	return sanitized or "config"


def _default_config_name(config: Mapping[str, Any], config_path: Path, override: str | None = None) -> str:
	if override not in (None, "", "null"):
		return _sanitize_config_name(override)
	experiment = config.get("experiment", {}) if isinstance(config.get("experiment"), Mapping) else {}
	name = experiment.get("name")
	if name not in (None, "", "null"):
		return _sanitize_config_name(name)
	return _sanitize_config_name(config_path.stem)


def _normalization_output_dir(config_path: Path, config: Mapping[str, Any], normalization_config: Mapping[str, Any], override: str | None = None) -> Path:
	if override not in (None, "", "null"):
		return _resolve_path(config_path, str(override))
	configured = normalization_config.get("output_dir")
	if configured not in (None, "", "null"):
		return _resolve_path(config_path, str(configured))
	paths = config.get("paths", {}) if isinstance(config.get("paths"), Mapping) else {}
	normalization_root = paths.get("normalization_root")
	if normalization_root not in (None, "", "null"):
		return _resolve_path(config_path, str(normalization_root))
	fallback = normalization_config.get("stats_path", normalization_config.get("path"))
	if fallback in (None, "", "null"):
		raise KeyError("Config is missing normalization.output_dir, paths.normalization_root, and normalization.stats_path/path.")
	return _resolve_path(config_path, str(fallback)).parent


def _jsonable(value: Any) -> Any:
	if isinstance(value, np.ndarray):
		if value.ndim == 0:
			return value.item()
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, Mapping):
		return {str(key): _jsonable(nested) for key, nested in value.items()}
	if isinstance(value, (list, tuple)):
		return [_jsonable(item) for item in value]
	return value


def _sha256_or_none(path: Path | None) -> str | None:
	if path is None or not path.exists() or not path.is_file():
		return None
	return compute_file_sha256(path)


def _npz_safe_metadata_value(value: Any) -> np.ndarray | None:
	if value is None:
		return None
	if isinstance(value, (str, int, float, bool, np.number, np.bool_)):
		return np.asarray(value)
	if isinstance(value, (list, tuple)):
		array = np.asarray(value)
		return None if array.dtype == object else array
	if isinstance(value, np.ndarray):
		return None if value.dtype == object else value
	return None


def _update_latest_alias(source: Path, alias: Path, *, as_copy: bool) -> None:
	alias.parent.mkdir(parents=True, exist_ok=True)
	alias.unlink(missing_ok=True)
	if as_copy:
		shutil.copy2(source, alias)
		return
	try:
		alias.symlink_to(source)
	except OSError:
		shutil.copy2(source, alias)


def _save_stats(
	config_path: Path,
	config: Mapping[str, Any],
	normalization_config: Mapping[str, Any],
	stats: dict[str, np.ndarray],
	metadata: Mapping[str, Any] | None = None,
	output_dir: str | None = None,
	config_name: str | None = None,
	update_latest_aliases: bool = True,
	latest_as_copy: bool = False,
) -> dict[str, Path]:
	"""Save timestamped normalization stats and metadata."""

	created_at = datetime.now().astimezone()
	timestamp = created_at.strftime("%Y%m%d_%H%M%S")
	resolved_config_name = _default_config_name(config, config_path, config_name)
	resolved_output_dir = _normalization_output_dir(config_path, config, normalization_config, output_dir)
	resolved_output_dir.mkdir(parents=True, exist_ok=True)
	npz_path = resolved_output_dir / f"train_normalization_stats_{resolved_config_name}_{timestamp}.npz"
	json_path = resolved_output_dir / f"train_normalization_stats_{resolved_config_name}_{timestamp}.json"
	latest_json_config = normalization_config.get("stats_path", normalization_config.get("path"))
	latest_json_path = _resolve_path(config_path, latest_json_config) if latest_json_config not in (None, "", "null") else resolved_output_dir / "normalization_stats.json"
	latest_npz_config = normalization_config.get("npz_path")
	latest_npz_path = _resolve_path(config_path, latest_npz_config) if latest_npz_config not in (None, "", "null") else latest_json_path.with_suffix(".npz")

	save_payload = dict(stats)
	for key, value in dict(metadata or {}).items():
		array_value = _npz_safe_metadata_value(value)
		if array_value is not None:
			save_payload[str(key)] = array_value
	np.savez_compressed(npz_path, **save_payload)

	config_path_absolute = config_path.expanduser().resolve()
	dataset_index_path = resolve_dataset_index_path(config)
	cache_dir = get_patch_cache_dir(config)
	cache_manifest_path = cache_dir / "cache_manifest.json"
	cache_manifest_hash = _sha256_or_none(cache_manifest_path)
	cache_config = config.get("cache", {}) if isinstance(config.get("cache"), Mapping) else {}
	resolved_config_text = json.dumps(_jsonable(dict(config)), sort_keys=True, default=str)
	metadata_payload = dict(metadata or {})
	json_payload = {
		"normalization_version": NORMALIZATION_VERSION,
		"created_at": created_at.isoformat(),
		"timestamp": timestamp,
		"config": {
			"config_name": resolved_config_name,
			"config_path": str(config_path),
			"config_path_absolute": str(config_path_absolute),
			"config_sha256": compute_file_sha256(config_path_absolute) if config_path_absolute.exists() else None,
			"resolved_config_sha256": compute_text_sha256(resolved_config_text),
			"base_config": config.get("_base_config_path", config.get("base_config")),
			"base_config_sha256": config.get("_base_config_sha256"),
		},
		"paths": {
			"output_dir": str(resolved_output_dir),
			"json_path": str(json_path),
			"npz_path": str(npz_path),
			"latest_json_path": str(latest_json_path),
			"latest_npz_path": str(latest_npz_path),
			"cache_dir": str(cache_dir),
			"dataset_index": str(dataset_index_path) if dataset_index_path is not None else None,
			"dataset_index_hash": compute_dataset_index_hash(config),
		},
		"data": {
			"fit_split": "train",
			"apply_to_splits": normalization_config.get("apply_to_splits", ["train", "val", "test"]),
			"input_sequence_length": int(config["input_sequence_length"]),
			"prediction_horizon": int(config["prediction_horizon"]),
			"input_channels": int(metadata_payload.get("input_channel_count", np.asarray(stats["mean"]).reshape(-1).shape[0])),
			"num_samples_used": int(metadata_payload.get("sample_count", 0)),
			"pixel_count": int(metadata_payload.get("pixel_count", 0)),
		},
		"cache": {
			"cache_version": str(cache_config.get("cache_version", "")),
			"cache_manifest_path": str(cache_manifest_path) if cache_manifest_path.exists() else None,
			"cache_manifest_hash": cache_manifest_hash,
			"cache_manifest_config_hash": metadata_payload.get("cache_manifest_config_hash"),
			"cache_manifest_dataset_index_hash": metadata_payload.get("cache_manifest_dataset_index_hash"),
		},
		"stats": {
			"format": "npz",
			"mean_key": "mean",
			"std_key": "std",
			"min_key": "min" if "min" in stats else None,
			"max_key": "max" if "max" in stats else None,
			"count_key": "count",
			"num_channels": int(np.asarray(stats["mean"]).reshape(-1).shape[0]),
			"npz_sha256": compute_file_sha256(npz_path),
		},
		"legacy_metadata": _jsonable(metadata_payload),
	}
	json_path.write_text(json.dumps(_jsonable(json_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
	if update_latest_aliases:
		_update_latest_alias(json_path, latest_json_path, as_copy=latest_as_copy)
		_update_latest_alias(npz_path, latest_npz_path, as_copy=latest_as_copy)
	return {
		"json_path": json_path,
		"npz_path": npz_path,
		"latest_json_path": latest_json_path,
		"latest_npz_path": latest_npz_path,
		"output_dir": resolved_output_dir,
	}


def _stats_metadata(config: Mapping[str, Any], *, mode: str, sample_count: int, pixel_count: int, input_channels: int) -> dict[str, Any]:
	"""Build provenance metadata for train-only normalization stats."""

	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = None if config_path_value in (None, "", "null") else Path(str(config_path_value)).expanduser().resolve()
	dataset_index_path = resolve_dataset_index_path(config)
	cache_dir = get_patch_cache_dir(config)
	cache_manifest_path = cache_dir / "cache_manifest.json"
	cache_manifest: dict[str, Any] = {}
	if cache_manifest_path.exists():
		try:
			cache_manifest = load_cache_manifest(cache_dir)
		except Exception:
			cache_manifest = {}
	resolved_config_text = json.dumps(config, sort_keys=True, default=str)
	return {
		"created_at_utc": datetime.now(timezone.utc).isoformat(),
		"fit_split": "train",
		"split_used": "train",
		"normalization_mode": mode,
		"normalization_method": str(resolve_normalization_config(config).get("method", "zscore")),
		"sample_count": int(sample_count),
		"num_samples_used": int(sample_count),
		"pixel_count": int(pixel_count),
		"count": int(pixel_count),
		"input_channel_count": int(input_channels),
		"input_sequence_length": int(config["input_sequence_length"]),
		"prediction_horizon": int(config["prediction_horizon"]),
		"config_path": str(config_path) if config_path is not None else "",
		"config_sha256": compute_file_sha256(config_path) if config_path is not None and config_path.exists() else None,
		"resolved_config_sha256": compute_text_sha256(resolved_config_text),
		"base_config_path": config.get("_base_config_path", config.get("base_config")),
		"base_config_sha256": config.get("_base_config_sha256"),
		"dataset_index_path": str(dataset_index_path) if dataset_index_path is not None else None,
		"dataset_index_hash": compute_dataset_index_hash(config),
		"cache_dir": str(cache_dir),
		"cache_version": str(dict(config.get("cache", {})).get("cache_version", "")) if isinstance(config.get("cache"), Mapping) else "",
		"cache_manifest_path": str(cache_manifest_path) if cache_manifest_path.exists() else None,
		"cache_manifest_config_hash": cache_manifest.get("config_hash"),
		"cache_manifest_dataset_index_hash": cache_manifest.get("dataset_index_hash"),
	}


def main() -> None:
	"""Compute and save normalization statistics for the training split only."""

	args = build_arg_parser().parse_args()
	config_path = Path(args.config).expanduser().resolve()
	config = load_config(config_path)
	config["config_path"] = str(config_path)

	for required_key in ("data_dir", "file_pattern", "input_sequence_length", "prediction_horizon", "train_fraction", "val_fraction"):
		if required_key not in config:
			raise KeyError(f"Config is missing required key '{required_key}'.")

	normalization_config = resolve_normalization_config(config)
	if "path" not in normalization_config:
		raise KeyError("Config is missing normalization.path.")

	compute_from_cache = bool(args.from_cache or normalization_config.get("compute_from_patch_cache", False))
	if compute_from_cache:
		from src.data.cache import validate_patch_cache
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
		metadata = _stats_metadata(
			config,
			mode="patch_cache",
			sample_count=len(train_dataset),
			pixel_count=count,
			input_channels=train_dataset.total_input_channels,
		)
		saved_paths = _save_stats(
			config_path,
			config,
			normalization_config,
			stats,
			metadata,
			output_dir=args.output_dir,
			config_name=args.config_name,
			update_latest_aliases=not bool(args.no_latest_alias),
			latest_as_copy=bool(args.latest_as_copy),
		)
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
		print("Saved normalization JSON:")
		print(f"  {saved_paths['json_path']}")
		print("Saved normalization NPZ:")
		print(f"  {saved_paths['npz_path']}")
		if not bool(args.no_latest_alias):
			print("Updated latest aliases:")
			print(f"  {saved_paths['latest_json_path']}")
			print(f"  {saved_paths['latest_npz_path']}")
		print("Suggested config:")
		print("normalization:")
		print(f"  output_dir: {saved_paths['output_dir']}")
		print(f"  stats_path: {saved_paths['latest_json_path']}")
		print(f"  npz_path: {saved_paths['latest_npz_path']}")
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

	metadata = _stats_metadata(
		config,
		mode=str(split_mode),
		sample_count=len(train_dataset),
		pixel_count=count,
		input_channels=train_dataset.total_input_channels,
	)
	saved_paths = _save_stats(
		config_path,
		config,
		normalization_config,
		stats,
		metadata,
		output_dir=args.output_dir,
		config_name=args.config_name,
		update_latest_aliases=not bool(args.no_latest_alias),
		latest_as_copy=bool(args.latest_as_copy),
	)

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
	print("Saved normalization JSON:")
	print(f"  {saved_paths['json_path']}")
	print("Saved normalization NPZ:")
	print(f"  {saved_paths['npz_path']}")
	if not bool(args.no_latest_alias):
		print("Updated latest aliases:")
		print(f"  {saved_paths['latest_json_path']}")
		print(f"  {saved_paths['latest_npz_path']}")
	print("Suggested config:")
	print("normalization:")
	print(f"  output_dir: {saved_paths['output_dir']}")
	print(f"  stats_path: {saved_paths['latest_json_path']}")
	print(f"  npz_path: {saved_paths['latest_npz_path']}")


if __name__ == "__main__":
	main()
