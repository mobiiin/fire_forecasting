"""Patch-cache helpers for precomputed wildfire training samples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.patching import resolve_patching_config, resolve_split_patch_mode, resolve_split_patch_stride
from src.data.stored_npz import stored_npz_array_info


DEFAULT_CACHE_DIR = Path("/scratch/mhabibp/fire_forecasting_patch_cache")
MANIFEST_FILENAME = "cache_manifest.json"
TARGET_DEFINITION_VERSION = "interval_consumed_current_to_horizon_target_v1"


def _get_section(config: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
	if not isinstance(config, Mapping):
		return {}
	section = config.get(name)
	return dict(section) if isinstance(section, Mapping) else {}


def target_definition_version(config: Mapping[str, Any] | None = None) -> str:
	"""Return the semantic version for how y is built from current/target frames."""

	if isinstance(config, Mapping):
		cache_config = _get_section(config, "cache")
		configured = cache_config.get("target_definition_version", config.get("target_definition_version"))
		if configured not in (None, "", "null"):
			return str(configured)
	return TARGET_DEFINITION_VERSION


def temporal_target_offsets(config: Mapping[str, Any]) -> dict[str, int]:
	"""Return target offsets from sample start and from the last input frame."""

	input_sequence_length = int(config["input_sequence_length"])
	prediction_horizon = int(config["prediction_horizon"])
	return {
		"target_offset_from_start": input_sequence_length - 1 + prediction_horizon,
		"target_offset_from_last_input": prediction_horizon,
	}


def _resolve_path(config: Mapping[str, Any], configured_path: str | Path) -> Path:
	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	config_path_value = config.get("config_path", config.get("_config_path"))
	if config_path_value:
		return (Path(config_path_value).expanduser().resolve().parent / path).resolve()
	return path.resolve()


def get_patch_cache_dir(config: Mapping[str, Any]) -> Path:
	"""Resolve the configured patch-cache directory."""

	cache_config = _get_section(config, "cache")
	cache_dir = cache_config.get("cache_dir")
	if cache_dir not in (None, "", "null"):
		return _resolve_path(config, cache_dir)
	scratch_root = cache_config.get("scratch_root")
	if scratch_root not in (None, "", "null"):
		return _resolve_path(config, scratch_root) / "fire_forecasting_patch_cache"
	return DEFAULT_CACHE_DIR


def load_cache_manifest(cache_dir: str | Path) -> dict[str, Any]:
	"""Load a patch-cache manifest from a cache directory."""

	manifest_path = Path(cache_dir).expanduser().resolve() / MANIFEST_FILENAME
	if not manifest_path.exists():
		raise FileNotFoundError(f"Patch-cache manifest not found: {manifest_path}")
	with manifest_path.open("r", encoding="utf-8") as handle:
		manifest = json.load(handle)
	if not isinstance(manifest, dict):
		raise ValueError(f"Patch-cache manifest must contain a JSON object: {manifest_path}")
	return manifest


def _jsonable(value: Any) -> Any:
	"""Convert config values into stable JSON-compatible objects."""

	if isinstance(value, Mapping):
		return {str(key): _jsonable(value[key]) for key in sorted(value)}
	if isinstance(value, (list, tuple)):
		return [_jsonable(item) for item in value]
	if isinstance(value, set):
		return sorted(_jsonable(item) for item in value)
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, np.ndarray):
		return value.tolist()
	if isinstance(value, np.generic):
		return value.item()
	return value


def compute_cache_config_hash(config: Mapping[str, Any]) -> str:
	"""Hash only config sections that affect precomputed patch contents."""

	relevant_keys = (
		"main_data_dir",
		"data_dir",
		"data_dirs",
		"fire_dataset_index_json",
		"temporal_trim",
		"data_discovery",
		"fire_filter",
		"manual_fire_split",
		"file_pattern",
		"input_sequence_length",
		"prediction_horizon",
		"target_channel",
		"task_type",
		"fire_threshold",
		"active_threshold",
		"input_channel_count",
		"input_channel_indices",
		"channel_layout",
		"engineered_features",
		"atmospheric_features",
		"energy_release",
		"geometry",
		"patching",
		"target_normalization",
	)
	payload = {key: config.get(key) for key in relevant_keys if key in config}
	model_section = _get_section(config, "model")
	payload["model"] = {
		"input_channels": model_section.get("input_channels"),
		"output_channels": model_section.get("output_channels"),
	}
	cache_section = _get_section(config, "cache")
	payload["cache"] = {
		"save_normalized_inputs": bool(cache_section.get("save_normalized_inputs", False)),
		"target_definition_version": target_definition_version(config),
	}
	multitask_section = _get_section(config, "multitask")
	payload["multitask"] = {
		key: multitask_section.get(key)
		for key in (
			"output_mode",
			"output_channels",
			"surface_fuel_channel",
			"canopy_fuel_channel",
			"flux_mask_channel",
			"mask_target_type",
			"flux_fire_threshold",
			"consumed_fuel_threshold",
			"clamp_consumed_fuel_targets_nonnegative",
		)
		if key in multitask_section
	}
	encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), default=str)
	return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resolve_config_relative_path(config: Mapping[str, Any], configured_path: str | Path) -> Path:
	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	config_path_value = config.get("config_path", config.get("_config_path"))
	if config_path_value:
		return (Path(config_path_value).expanduser().resolve().parent / path).resolve()
	return path.resolve()


def resolve_dataset_index_path(config: Mapping[str, Any]) -> Path | None:
	"""Resolve the configured fire dataset index path, if one is configured."""

	index_path = config.get("fire_dataset_index_json")
	if index_path in (None, "", "null"):
		return None
	return _resolve_config_relative_path(config, index_path)


def _sha256_file(path: Path) -> str:
	hasher = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			hasher.update(chunk)
	return hasher.hexdigest()


def compute_dataset_index_hash(config: Mapping[str, Any]) -> str | None:
	"""Hash the configured dataset index file so cache validation sees index edits."""

	index_path = resolve_dataset_index_path(config)
	if index_path is None or not index_path.exists():
		return None
	return _sha256_file(index_path)


def extract_temporal_trim_manifest(dataset_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
	"""Build compact per-fire trim metadata for cache manifests."""

	fires: dict[str, dict[str, Any]] = {}
	enabled = False
	for record in dataset_records:
		trim = dict(record.get("temporal_trim", {})) if isinstance(record.get("temporal_trim"), Mapping) else {}
		fire_name = str(record.get("dataset_name", record.get("data_dir", "")))
		fire_enabled = bool(trim.get("enabled", False))
		enabled = enabled or fire_enabled
		fires[fire_name] = {
			"enabled": fire_enabled,
			"trim_start_index": int(trim.get("trim_start_index", 0)),
			"trim_end_index": int(trim.get("trim_end_index", max(int(record.get("num_files", 1)) - 1, 0))),
			"trimmed_num_frames": int(trim.get("trimmed_num_frames", record.get("effective_num_files", record.get("num_files", 0)))),
			"original_num_frames": int(trim.get("original_num_frames", record.get("num_files", 0))),
		}
	return {"temporal_trim_enabled": enabled, "fires": fires}


def compute_trim_metadata_hash(dataset_records: Sequence[Mapping[str, Any]]) -> str:
	"""Hash compact trim metadata from discovered dataset records."""

	payload = extract_temporal_trim_manifest(dataset_records)
	encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), default=str)
	return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compute_current_trim_metadata_hash(config: Mapping[str, Any]) -> str | None:
	"""Hash temporal_trim metadata from the configured dataset index file."""

	index_path = resolve_dataset_index_path(config)
	if index_path is None or not index_path.exists():
		return None
	with index_path.open("r", encoding="utf-8") as handle:
		index = json.load(handle)
	fires = index.get("fires", {}) if isinstance(index, Mapping) else {}
	if not isinstance(fires, Mapping):
		return None
	payload: dict[str, Any] = {"temporal_trim_enabled": False, "fires": {}}
	for fire_name, record in sorted(fires.items()):
		if not isinstance(record, Mapping):
			continue
		trim = dict(record.get("temporal_trim", {})) if isinstance(record.get("temporal_trim"), Mapping) else {}
		enabled = bool(trim.get("enabled", False))
		payload["temporal_trim_enabled"] = bool(payload["temporal_trim_enabled"] or enabled)
		payload["fires"][str(fire_name)] = {
			"enabled": enabled,
			"trim_start_index": trim.get("trim_start_index"),
			"trim_end_index": trim.get("trim_end_index"),
			"trimmed_num_frames": trim.get("trimmed_num_frames"),
			"original_num_frames": trim.get("original_num_frames"),
		}
	encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), default=str)
	return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _compute_legacy_cache_config_hash(config: Mapping[str, Any]) -> str:
	"""Return the historical hash used by existing patch-cache manifests."""

	relevant_keys = (
		"main_data_dir",
		"data_dir",
		"data_dirs",
		"fire_dataset_index_json",
		"data_discovery",
		"fire_filter",
		"manual_fire_split",
		"file_pattern",
		"input_sequence_length",
		"prediction_horizon",
		"target_channel",
		"task_type",
		"fire_threshold",
		"active_threshold",
		"input_channel_count",
		"input_channel_indices",
		"channel_layout",
		"engineered_features",
		"atmospheric_features",
		"multitask",
		"energy_release",
		"geometry",
		"patching",
		"target_normalization",
	)
	payload = {key: config.get(key) for key in relevant_keys if key in config}
	model_section = _get_section(config, "model")
	payload["model"] = {
		"input_channels": model_section.get("input_channels"),
		"output_channels": model_section.get("output_channels"),
	}
	cache_section = _get_section(config, "cache")
	payload["cache"] = {
		"save_normalized_inputs": bool(cache_section.get("save_normalized_inputs", False)),
	}
	encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), default=str)
	return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _required_splits(split: str | Sequence[str] | None) -> list[str]:
	if split is None:
		return ["train", "val", "test"]
	if isinstance(split, str):
		if split.lower() == "all":
			return ["train", "val", "test"]
		return [split.lower()]
	return [str(item).lower() for item in split]


def _shard_path(cache_dir: Path, shard_entry: Mapping[str, Any] | str) -> Path:
	if isinstance(shard_entry, str):
		path_value = shard_entry
	else:
		path_value = shard_entry.get("path")
	if path_value in (None, "", "null"):
		raise ValueError(f"Shard entry is missing a path: {shard_entry!r}")
	path = Path(str(path_value)).expanduser()
	if path.is_absolute():
		return path.resolve()
	return (cache_dir / path).resolve()


def _read_shard_shapes(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
	if path.suffix.lower() == ".npz":
		try:
			return stored_npz_array_info(path, "X").shape, stored_npz_array_info(path, "y").shape
		except (KeyError, ValueError):
			with np.load(path, allow_pickle=False) as shard:
				if "X" not in shard.files or "y" not in shard.files:
					raise ValueError(f"Shard is missing required X/y arrays: {path}")
				return tuple(int(value) for value in shard["X"].shape), tuple(int(value) for value in shard["y"].shape)
	if path.suffix.lower() == ".pt":
		try:
			import torch  # type: ignore[import-not-found]
		except ImportError as exc:  # pragma: no cover - optional format
			raise ImportError("PyTorch is required to validate .pt patch-cache shards.") from exc
		shard = torch.load(path, map_location="cpu")
		if "X" not in shard or "y" not in shard:
			raise ValueError(f"Shard is missing required X/y tensors: {path}")
		return tuple(int(value) for value in shard["X"].shape), tuple(int(value) for value in shard["y"].shape)
	raise ValueError(f"Unsupported patch-cache shard format: {path}")


def _cache_missing_message(cache_dir: Path, split_text: str | None = None) -> str:
	split_arg = split_text if split_text is not None else "all"
	return (
		f"Precomputed patch cache is missing or incomplete under {cache_dir}.\n"
		"Scratch files may have been deleted.\n"
		"Please rerun:\n"
		f"python scripts/precompute_patch_cache.py --config configs/default.yaml --split {split_arg}"
	)


def _patch_settings_mismatch_message(cache_dir: Path, manifest: Mapping[str, Any], config: Mapping[str, Any]) -> str:
	manifest_modes = manifest.get("patch_modes", {})
	manifest_strides = manifest.get("strides", {})
	expected_modes = {split: resolve_split_patch_mode(config, split) for split in ("train", "val", "test")}
	expected_strides = {split: resolve_split_patch_stride(config, split) for split in ("train", "val", "test")}
	return (
		"Patch cache was built with different patch settings.\n"
		f"Manifest patch_modes={manifest_modes} strides={manifest_strides}\n"
		f"Expected patch_modes={expected_modes} strides={expected_strides}\n"
		"Expected train/val/test patch_mode=sliding_window and stride=60.\n"
		"Please rerun:\n"
		"python scripts/precompute_patch_cache.py --config configs/default.yaml --split all"
	)


def validate_patch_cache(config: Mapping[str, Any], split: str | Sequence[str] | None = None) -> dict[str, Any]:
	"""Validate a precomputed patch cache and return a compact summary."""

	cache_config = _get_section(config, "cache")
	cache_dir = get_patch_cache_dir(config)
	requested_splits = _required_splits(split)
	if not cache_dir.exists():
		raise RuntimeError(_cache_missing_message(cache_dir, requested_splits[0] if len(requested_splits) == 1 else None))
	manifest_path = cache_dir / MANIFEST_FILENAME
	if not manifest_path.exists():
		raise RuntimeError(_cache_missing_message(cache_dir, requested_splits[0] if len(requested_splits) == 1 else None))
	manifest = load_cache_manifest(cache_dir)

	expected_version = str(cache_config.get("cache_version", "v1"))
	actual_version = str(manifest.get("cache_version", ""))
	if actual_version != expected_version:
		raise RuntimeError(
			f"Patch-cache version mismatch under {cache_dir}: manifest has {actual_version!r}, "
			f"config expects {expected_version!r}. Rerun precompute or update cache.cache_version."
		)

	expected_hash = compute_cache_config_hash(config)
	actual_hash = str(manifest.get("config_hash", ""))
	legacy_hash = _compute_legacy_cache_config_hash(config)
	hash_status = "current" if actual_hash == expected_hash else "legacy" if actual_hash == legacy_hash else "mismatch"
	if hash_status == "mismatch" and not bool(cache_config.get("allow_config_hash_mismatch", False)):
		raise RuntimeError(
			f"Patch-cache config hash mismatch under {cache_dir}.\n"
			f"manifest config_hash={actual_hash}\n"
			f"current  config_hash={expected_hash}\n"
			f"legacy  config_hash={legacy_hash}\n"
			"Rerun precompute, or set cache.allow_config_hash_mismatch=true only if the cache-affecting config change is intentional."
		)

	expected_index_hash = compute_dataset_index_hash(config)
	actual_index_hash = manifest.get("dataset_index_hash")
	if actual_index_hash not in (None, "", "null") and expected_index_hash not in (None, "", "null"):
		if str(actual_index_hash) != str(expected_index_hash):
			index_path = resolve_dataset_index_path(config)
			raise RuntimeError(
				f"Patch-cache dataset index hash mismatch under {cache_dir}.\n"
				f"manifest dataset_index_hash={actual_index_hash}\n"
				f"current  dataset_index_hash={expected_index_hash}\n"
				f"index={index_path}\n"
				"Rerun:\n"
				"python scripts/precompute_patch_cache.py --config configs/default.yaml --split all --overwrite"
			)

	expected_trim_hash = compute_current_trim_metadata_hash(config)
	actual_trim_hash = manifest.get("trim_metadata_hash")
	if actual_trim_hash not in (None, "", "null") and expected_trim_hash not in (None, "", "null"):
		if str(actual_trim_hash) != str(expected_trim_hash):
			raise RuntimeError(
				f"Patch-cache temporal trim metadata hash mismatch under {cache_dir}.\n"
				f"manifest trim_metadata_hash={actual_trim_hash}\n"
				f"current  trim_metadata_hash={expected_trim_hash}\n"
				"Rerun:\n"
				"python scripts/precompute_patch_cache.py --config configs/default.yaml --split all --overwrite"
			)

	patching = resolve_patching_config(config)
	expected_t = int(config["input_sequence_length"])
	expected_horizon = int(config["prediction_horizon"])
	expected_c = int(_get_section(config, "model").get("input_channels", manifest.get("input_channels", 0)))
	expected_yc = int(_get_section(config, "model").get("output_channels", manifest.get("output_channels", 1)))
	expected_h = int(patching["patch_height"])
	expected_w = int(patching["patch_width"])
	expected_target_definition = target_definition_version(config)
	expected_offsets = temporal_target_offsets(config)
	for manifest_key, expected_value in (
		("input_sequence_length", expected_t),
		("prediction_horizon", expected_horizon),
		("target_offset_from_start", expected_offsets["target_offset_from_start"]),
		("target_offset_from_last_input", expected_offsets["target_offset_from_last_input"]),
	):
		if manifest_key not in manifest:
			raise RuntimeError(
				f"Patch-cache manifest is missing {manifest_key!r}; old caches must be rebuilt for the current sequence definition."
			)
		if int(manifest.get(manifest_key)) != int(expected_value):
			raise RuntimeError(
				f"Patch-cache {manifest_key} mismatch under {cache_dir}: "
				f"manifest={manifest.get(manifest_key)!r}, config={expected_value!r}. Rerun precompute."
			)
	actual_target_definition = str(manifest.get("target_definition_version", ""))
	if actual_target_definition != expected_target_definition:
		raise RuntimeError(
			f"Patch-cache target_definition_version mismatch under {cache_dir}: "
			f"manifest={actual_target_definition!r}, config={expected_target_definition!r}. Rerun precompute."
		)
	if expected_h != int(manifest.get("patch_height", expected_h)) or expected_w != int(manifest.get("patch_width", expected_w)):
		raise RuntimeError(
			"Patch-cache spatial shape does not match config: "
			f"manifest=({manifest.get('patch_height')}, {manifest.get('patch_width')}) "
			f"config=({expected_h}, {expected_w})."
		)
	manifest_patch_modes = manifest.get("patch_modes", {})
	manifest_strides = manifest.get("strides", {})
	if not isinstance(manifest_patch_modes, Mapping) or not isinstance(manifest_strides, Mapping):
		raise RuntimeError(_patch_settings_mismatch_message(cache_dir, manifest, config))

	shards_by_split = manifest.get("shards")
	if not isinstance(shards_by_split, Mapping):
		raise RuntimeError(f"Patch-cache manifest is missing a 'shards' mapping: {manifest_path}")
	summaries: dict[str, dict[str, Any]] = {}
	for split_name in requested_splits:
		expected_patch_mode = resolve_split_patch_mode(config, split_name)
		expected_stride = resolve_split_patch_stride(config, split_name)
		actual_patch_mode = str(manifest_patch_modes.get(split_name, ""))
		actual_stride = int(manifest_strides.get(split_name, -1))
		if actual_patch_mode != expected_patch_mode or actual_stride != expected_stride:
			raise RuntimeError(_patch_settings_mismatch_message(cache_dir, manifest, config))
		split_dir = cache_dir / split_name
		if not split_dir.exists():
			raise RuntimeError(_cache_missing_message(cache_dir, split_name))
		shard_entries = shards_by_split.get(split_name, [])
		if not isinstance(shard_entries, Sequence) or isinstance(shard_entries, (str, bytes)) or not shard_entries:
			raise RuntimeError(_cache_missing_message(cache_dir, split_name))
		shard_paths = [_shard_path(cache_dir, entry) for entry in shard_entries]
		missing = [path for path in shard_paths if not path.exists()]
		if missing:
			raise RuntimeError(_cache_missing_message(cache_dir, split_name) + "\nMissing shards:\n" + "\n".join(str(path) for path in missing[:10]))
		first_shape_x, first_shape_y = _read_shard_shapes(shard_paths[0])
		if len(first_shape_x) != 5:
			raise RuntimeError(f"Expected shard X shape (N,T,C,H,W), got {first_shape_x} in {shard_paths[0]}")
		if len(first_shape_y) != 4:
			raise RuntimeError(f"Expected shard y shape (N,C,H,W), got {first_shape_y} in {shard_paths[0]}")
		if first_shape_x[1:] != (expected_t, expected_c, expected_h, expected_w):
			raise RuntimeError(
				f"Patch-cache X shape mismatch in {shard_paths[0]}: "
				f"got {first_shape_x}, expected (N,{expected_t},{expected_c},{expected_h},{expected_w})."
			)
		if first_shape_y[1:] != (expected_yc, expected_h, expected_w):
			raise RuntimeError(
				f"Patch-cache y shape mismatch in {shard_paths[0]}: "
				f"got {first_shape_y}, expected (N,{expected_yc},{expected_h},{expected_w})."
			)
		num_samples = int(manifest.get(f"num_{split_name}_patches", 0))
		if num_samples <= 0:
			num_samples = sum(int(entry.get("num_samples", 0)) if isinstance(entry, Mapping) else 0 for entry in shard_entries)
		if num_samples <= 0:
			raise RuntimeError(f"Patch-cache manifest reports zero samples for split={split_name!r}.")
		summaries[split_name] = {
			"num_samples": num_samples,
			"num_shards": len(shard_paths),
			"first_shard": str(shard_paths[0]),
			"x_shape": first_shape_x,
			"y_shape": first_shape_y,
			"patch_mode": actual_patch_mode,
			"stride": actual_stride,
		}

	return {
		"cache_dir": str(cache_dir),
		"manifest_path": str(manifest_path),
		"manifest": manifest,
		"splits": summaries,
		"config_hash": expected_hash,
		"legacy_config_hash": legacy_hash,
		"manifest_config_hash": actual_hash,
		"config_hash_status": hash_status,
	}
