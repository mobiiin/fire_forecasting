"""PyTorch dataset and DataLoader helpers for sequence-to-map wildfire forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
	import torch  # type: ignore[import-not-found]
	from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	DataLoader = None
	WeightedRandomSampler = None

	class Dataset:  # type: ignore[too-many-ancestors]
		"""Fallback base class used only when PyTorch is unavailable."""

		pass

from src.data.discovery import discover_dataset_files, discover_multiple_datasets, resolve_data_dirs, sort_chronologically
from src.data.cache import target_definition_version, temporal_target_offsets
from src.data.energy_release import (
	compute_energy_release_maps,
	resolve_energy_output_channel_names,
	resolve_energy_release_config,
	resolve_energy_target_count,
	transform_energy_target,
)
from src.data.geometry import load_fire_geometry
from src.data.patching import (
	extract_patch_array,
	resolve_patching_config,
	resolve_split_patch_mode,
	sample_active_patch,
	sample_random_patch,
	validate_patch_dict,
)
from src.data.processed_sample_dataset import ProcessedTemporalPatchDataset
from src.data.preprocessing import (
	input_normalization_runs_on_device,
	load_normalization_stats,
	normalize_channel_map,
	normalize_tensor,
)
from src.data.splits import (
	build_sliding_patch_refs_for_split,
	chronological_split_indices,
	chronological_train_val_split_indices,
	manual_fire_holdout_splits,
	multi_dataset_chronological_splits,
	multi_fire_chronological_splits,
)
from src.data.temporal_trim import effective_num_frames, resolve_temporal_trim, temporal_sample_metadata
from src.training.hardware import cap_num_workers_by_slurm, get_performance_config


def _extract_numeric_suffix(name: str) -> int | None:
	"""Extract a trailing numeric suffix from a filename stem if present."""

	digits = []
	for character in reversed(name):
		if character.isdigit():
			digits.append(character)
		else:
			break
	if not digits:
		return None
	return int("".join(reversed(digits)))


def _sort_chronologically(file_paths: Sequence[Path]) -> list[Path]:
	"""Sort by trailing numeric suffix when available, otherwise lexicographically."""

	return sort_chronologically(file_paths)


def _resolve_path(base_path: Path | None, configured_path: str | Path) -> Path:
	"""Resolve a configured path relative to a config file when available."""

	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	if base_path is None:
		return path.resolve()
	return (base_path.parent / path).resolve()


def _as_path_list(file_paths: Iterable[str | Path]) -> list[Path]:
	"""Convert an arbitrary iterable of paths into a concrete list of Path objects."""

	return [Path(path) for path in file_paths]


def _extract_first_metadata_item(value):
	"""Normalize one collated metadata field to its first scalar/string item."""

	if torch is not None and torch.is_tensor(value):
		return value.reshape(-1)[0].item() if value.numel() else None
	if isinstance(value, (list, tuple)):
		return value[0] if value else None
	return value


def _metadata_lengths(value: Any) -> tuple[list[int], list[int]]:
	"""Collect possible batch lengths from a collated metadata value."""

	tensor_lengths: list[int] = []
	sequence_lengths: list[int] = []
	if torch is not None and torch.is_tensor(value):
		if value.ndim > 0:
			tensor_lengths.append(int(value.shape[0]))
		return tensor_lengths, sequence_lengths
	if isinstance(value, Mapping):
		for nested_value in value.values():
			nested_tensor_lengths, nested_sequence_lengths = _metadata_lengths(nested_value)
			tensor_lengths.extend(nested_tensor_lengths)
			sequence_lengths.extend(nested_sequence_lengths)
		return tensor_lengths, sequence_lengths
	if isinstance(value, (list, tuple)):
		sequence_lengths.append(len(value))
	return tensor_lengths, sequence_lengths


def _most_likely_metadata_batch_size(lengths: Sequence[int]) -> int | None:
	"""Choose the most common positive candidate length, preferring the larger value on ties."""

	counts: dict[int, int] = {}
	for length in lengths:
		if int(length) > 0:
			counts[int(length)] = counts.get(int(length), 0) + 1
	if not counts:
		return None
	return max(counts, key=lambda length: (counts[length], length))


def _metadata_value_for_sample(value: Any, sample_index: int, batch_size: int) -> Any:
	"""Extract one sample's metadata value while preserving batch-level fields."""

	if torch is not None and torch.is_tensor(value):
		if value.ndim == 0:
			return value.item()
		if int(value.shape[0]) == int(batch_size):
			sample_value = value[sample_index]
			return sample_value.item() if sample_value.ndim == 0 else sample_value
		return value.detach().cpu().tolist()
	if isinstance(value, Mapping):
		return {key: _metadata_value_for_sample(nested_value, sample_index, batch_size) for key, nested_value in value.items()}
	if isinstance(value, (list, tuple)):
		if len(value) == int(batch_size):
			return value[sample_index]
		if len(value) == 1:
			return value[0]
		return list(value)
	return value


def metadata_batch_to_list(metadata_batch: Mapping[str, Any], batch_size: int | None = None) -> list[dict[str, Any]]:
	"""Convert a collated metadata batch into a list of per-sample dictionaries."""

	if not isinstance(metadata_batch, Mapping):
		raise TypeError(f"Expected metadata_batch to be a mapping, got {type(metadata_batch)!r}.")
	if not metadata_batch:
		return []

	resolved_batch_size = int(batch_size) if batch_size is not None else None
	if resolved_batch_size is None:
		tensor_lengths: list[int] = []
		sequence_lengths: list[int] = []
		for value in metadata_batch.values():
			value_tensor_lengths, value_sequence_lengths = _metadata_lengths(value)
			tensor_lengths.extend(value_tensor_lengths)
			sequence_lengths.extend(value_sequence_lengths)
		resolved_batch_size = _most_likely_metadata_batch_size(tensor_lengths) or _most_likely_metadata_batch_size(sequence_lengths)
	if resolved_batch_size is None:
		return [{key: _extract_first_metadata_item(value) for key, value in metadata_batch.items()}]

	items: list[dict[str, Any]] = []
	for sample_index in range(resolved_batch_size):
		item: dict[str, Any] = {}
		for key, value in metadata_batch.items():
			item[key] = _metadata_value_for_sample(value, sample_index, resolved_batch_size)
		items.append(item)
	return items


def _get_section(config: Mapping[str, Any] | None, *names: str) -> dict[str, Any]:
	"""Return the first nested mapping found under any of the provided names."""

	if not isinstance(config, Mapping):
		return {}
	for name in names:
		section = config.get(name)
		if isinstance(section, Mapping):
			return dict(section)
	return {}


def _coerce_index_list(value: Sequence[int] | None) -> list[int] | None:
	"""Convert an optional sequence of indices into a concrete list."""

	if value is None:
		return None
	return [int(item) for item in value]


def _resolve_input_channel_indices(config: Mapping[str, Any], input_channel_count: int) -> list[int]:
	"""Resolve the base input channels from config."""

	configured_indices = config.get("input_channel_indices")
	if configured_indices is None:
		return list(range(int(input_channel_count)))
	if not isinstance(configured_indices, Sequence):
		raise TypeError("input_channel_indices must be null or a sequence of integers.")
	resolved = [int(index) for index in configured_indices]
	if not resolved:
		raise ValueError("input_channel_indices cannot be empty when provided.")
	return resolved


def _resolve_channel_layout(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Resolve channel layout information from config."""

	layout = _get_section(config, "channel_layout")
	if not layout:
		raise KeyError("Config is missing channel_layout, which is required for engineered features and multitask targets.")

	flux_channels = _coerce_index_list(layout.get("flux_channels"))
	fuel_channels = _coerce_index_list(layout.get("fuel_channels"))
	if flux_channels is None or fuel_channels is None:
		raise KeyError("channel_layout must define flux_channels and fuel_channels.")
	if len(fuel_channels) != 2:
		raise ValueError(f"channel_layout.fuel_channels must contain exactly 2 channels, got {fuel_channels}.")

	surface_fuel_channel = int(layout.get("surface_fuel_channel", fuel_channels[0]))
	canopy_fuel_channel = int(layout.get("canopy_fuel_channel", fuel_channels[1]))
	flux_mask_channel = int(layout.get("flux_mask_channel", flux_channels[0]))

	return {
		"atmospheric_channels": layout.get("atmospheric_channels"),
		"flux_channels": flux_channels,
		"fuel_channels": fuel_channels,
		"surface_fuel_channel": surface_fuel_channel,
		"canopy_fuel_channel": canopy_fuel_channel,
		"flux_mask_channel": flux_mask_channel,
	}


def _resolve_engineered_features_config(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Resolve engineered-feature flags with defaults."""

	section = _get_section(config, "engineered_features")
	return {
		"enabled": bool(section.get("enabled", False)),
		"add_flux_delta": bool(section.get("add_flux_delta", False)),
		"add_fuel_delta": bool(section.get("add_fuel_delta", False)),
		"add_step_consumed_fuel": bool(section.get("add_step_consumed_fuel", False)),
		"add_cumulative_consumed_fuel": bool(section.get("add_cumulative_consumed_fuel", False)),
		"initial_fuel_mode": str(section.get("initial_fuel_mode", "first_dataset_frame")).lower(),
		"clamp_consumed_fuel_nonnegative": bool(section.get("clamp_consumed_fuel_nonnegative", True)),
	}


def _resolve_atmospheric_features_config(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Resolve atmospheric engineered-feature flags with defaults."""

	section = _get_section(config, "atmospheric_features")
	return {
		"enabled": bool(section.get("enabled", False)),
		"num_vertical_levels": int(section.get("num_vertical_levels", 8)),
		"variables_per_level": int(section.get("variables_per_level", 10)),
		"add_horizontal_wind_speed": bool(section.get("add_horizontal_wind_speed", False)),
		"add_low_level_mean_wind_speed": bool(section.get("add_low_level_mean_wind_speed", False)),
		"add_updraft": bool(section.get("add_updraft", False)),
		"add_wind_direction": bool(section.get("add_wind_direction", False)),
		"wind_direction_mode": str(section.get("wind_direction_mode", "unit_vector")).lower(),
		"wind_direction_convention": str(section.get("wind_direction_convention", "toward")).lower(),
		"low_level_indices": [int(index) for index in section.get("low_level_indices", [0, 1, 2])],
		"epsilon": float(section.get("epsilon", 1e-6)),
	}


def _resolve_multitask_config(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Resolve multitask config with channel-layout fallbacks."""

	layout = _resolve_channel_layout(config)
	multitask = _get_section(config, "multitask")
	if not multitask:
		raise KeyError("Config is missing multitask section for task_type='multitask'.")

	return {
		"output_mode": str(multitask.get("output_mode", "surface_canopy_consumed_plus_mask")),
		"surface_fuel_channel": int(multitask.get("surface_fuel_channel", layout["surface_fuel_channel"])),
		"canopy_fuel_channel": int(multitask.get("canopy_fuel_channel", layout["canopy_fuel_channel"])),
		"flux_mask_channel": int(multitask.get("flux_mask_channel", layout["flux_mask_channel"])),
		"mask_target_type": str(multitask.get("mask_target_type", "active_flux")).lower(),
		"flux_fire_threshold": float(multitask.get("flux_fire_threshold", 0.05)),
		"consumed_fuel_threshold": float(multitask.get("consumed_fuel_threshold", 0.01)),
		"clamp_consumed_fuel_targets_nonnegative": bool(multitask.get("clamp_consumed_fuel_targets_nonnegative", True)),
	}


def _multitask_output_channel_count(config: Mapping[str, Any]) -> int:
	"""Return the total multitask target channel count for the current config."""

	return 3 + resolve_energy_target_count(config)


def _resolve_target_normalization_config(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Resolve target normalization configuration."""

	section = _get_section(config, "target_normalization")
	if not section:
		section = _get_section(config, "normalization")
	return {
		"enabled": bool(section.get("enabled", section.get("normalize_target", False))),
		"method": str(section.get("method", "zscore")).lower(),
	}


def _resolve_square_patch_size_from_config(config: Mapping[str, Any]) -> int:
	"""Resolve one square patch size from patching config or legacy top-level keys."""

	patching = resolve_patching_config(config)
	patch_height = int(patching["patch_height"])
	patch_width = int(patching["patch_width"])
	if patch_height != patch_width:
		raise ValueError(
			"Current dataset classes require square patches. "
			f"Got patch_height={patch_height}, patch_width={patch_width}."
		)
	return int(patch_height)


def _count_fuel_flux_engineered_channels(config: Mapping[str, Any]) -> int:
	"""Count the number of fuel/flux engineered input channels that will be appended."""

	engineered = _resolve_engineered_features_config(config)
	if not engineered["enabled"]:
		return 0

	layout = _resolve_channel_layout(config)
	flux_count = len(layout["flux_channels"])
	fuel_count = len(layout["fuel_channels"])
	total = 0
	if engineered["add_flux_delta"]:
		total += flux_count
	if engineered["add_fuel_delta"]:
		total += fuel_count
	if engineered["add_step_consumed_fuel"]:
		total += fuel_count
	if engineered["add_cumulative_consumed_fuel"]:
		total += fuel_count
	return total


def count_atmospheric_engineered_channels(config: Mapping[str, Any]) -> int:
	"""Count the number of atmospheric engineered channels that will be appended."""

	atmospheric = _resolve_atmospheric_features_config(config)
	if not atmospheric["enabled"]:
		return 0

	num_vertical_levels = int(atmospheric["num_vertical_levels"])
	if atmospheric["add_wind_direction"]:
		if atmospheric["wind_direction_mode"] != "unit_vector":
			raise ValueError(
				"Unsupported atmospheric_features.wind_direction_mode. "
				f"Expected 'unit_vector', got {atmospheric['wind_direction_mode']!r}."
			)
		if atmospheric["wind_direction_convention"] != "toward":
			raise ValueError(
				"Unsupported atmospheric_features.wind_direction_convention. "
				f"Expected 'toward', got {atmospheric['wind_direction_convention']!r}."
			)
	total = 0
	if atmospheric["add_horizontal_wind_speed"]:
		total += num_vertical_levels
	if atmospheric["add_low_level_mean_wind_speed"]:
		total += 1
	if atmospheric["add_updraft"]:
		total += num_vertical_levels
	if atmospheric["add_wind_direction"]:
		total += 2 * num_vertical_levels
	return total


def _count_engineered_channels(config: Mapping[str, Any]) -> int:
	"""Count the total number of engineered input channels that will be appended."""

	energy_release = resolve_energy_release_config(config)
	energy_history_channels = 1 if energy_release["enabled"] and energy_release["add_as_input_history"] else 0
	return _count_fuel_flux_engineered_channels(config) + count_atmospheric_engineered_channels(config) + energy_history_channels


def _slice_channels(frame: np.ndarray, channel_indices: Sequence[int]) -> np.ndarray:
	"""Slice a frame with an explicit channel list."""

	return np.asarray(frame[:, :, list(channel_indices)], dtype=np.float32)


def build_atmospheric_features(frame: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
	"""Build atmospheric engineered features from one raw frame."""

	atmospheric = _resolve_atmospheric_features_config(config)
	if not atmospheric["enabled"]:
		if frame.ndim != 3:
			raise ValueError(f"build_atmospheric_features expects a raw frame shaped (H, W, C), got {frame.shape}.")
		height, width = int(frame.shape[0]), int(frame.shape[1])
		return np.zeros((height, width, 0), dtype=np.float32)

	raw_frame = np.asarray(frame, dtype=np.float32)
	if raw_frame.ndim != 3:
		raise ValueError(f"build_atmospheric_features expects a raw frame shaped (H, W, C), got {raw_frame.shape}.")

	num_vertical_levels = int(atmospheric["num_vertical_levels"])
	variables_per_level = int(atmospheric["variables_per_level"])
	required_raw_channels = num_vertical_levels * variables_per_level
	if num_vertical_levels <= 0:
		raise ValueError(f"atmospheric_features.num_vertical_levels must be positive, got {num_vertical_levels}.")
	if variables_per_level <= 0:
		raise ValueError(f"atmospheric_features.variables_per_level must be positive, got {variables_per_level}.")
	if required_raw_channels > raw_frame.shape[2]:
		raise ValueError(
			"Atmospheric engineered features require more raw channels than are available. "
			f"Need {required_raw_channels} from num_vertical_levels * variables_per_level, got {raw_frame.shape[2]}."
		)

	low_level_indices = [int(index) for index in atmospheric["low_level_indices"]]
	invalid_low_levels = [index for index in low_level_indices if index < 0 or index >= num_vertical_levels]
	if invalid_low_levels:
		raise ValueError(
			"atmospheric_features.low_level_indices contain invalid z-level indices. "
			f"Valid range is [0, {num_vertical_levels - 1}], got {invalid_low_levels}."
		)
	if atmospheric["add_low_level_mean_wind_speed"] and not low_level_indices:
		raise ValueError("atmospheric_features.low_level_indices cannot be empty when add_low_level_mean_wind_speed is enabled.")
	if atmospheric["add_wind_direction"]:
		if atmospheric["wind_direction_mode"] != "unit_vector":
			raise ValueError(
				"Unsupported atmospheric_features.wind_direction_mode. "
				f"Expected 'unit_vector', got {atmospheric['wind_direction_mode']!r}."
			)
		if atmospheric["wind_direction_convention"] != "toward":
			raise ValueError(
				"Unsupported atmospheric_features.wind_direction_convention. "
				f"Expected 'toward', got {atmospheric['wind_direction_convention']!r}."
			)

	epsilon = max(float(atmospheric["epsilon"]), 0.0)
	u_levels: list[np.ndarray] = []
	v_levels: list[np.ndarray] = []
	horizontal_wind_speed_levels: list[np.ndarray] = []
	updraft_levels: list[np.ndarray] = []

	for z_level in range(num_vertical_levels):
		base_index = z_level * variables_per_level
		u_values = np.asarray(raw_frame[:, :, base_index + 0], dtype=np.float32)
		v_values = np.asarray(raw_frame[:, :, base_index + 1], dtype=np.float32)
		w_values = np.asarray(raw_frame[:, :, base_index + 2], dtype=np.float32)
		u_levels.append(u_values)
		v_levels.append(v_values)
		if atmospheric["add_horizontal_wind_speed"]:
			wind_speed = np.sqrt(u_values * u_values + v_values * v_values + epsilon).astype(np.float32, copy=False)
			horizontal_wind_speed_levels.append(wind_speed[:, :, None].astype(np.float32, copy=False))
		if atmospheric["add_updraft"]:
			updraft = np.maximum(w_values, 0.0)
			updraft_levels.append(updraft[:, :, None].astype(np.float32, copy=False))

	feature_groups: list[np.ndarray] = []
	if atmospheric["add_horizontal_wind_speed"] and horizontal_wind_speed_levels:
		feature_groups.extend(horizontal_wind_speed_levels)

	if atmospheric["add_low_level_mean_wind_speed"]:
		selected_u = np.stack([u_levels[index] for index in low_level_indices], axis=0)
		selected_v = np.stack([v_levels[index] for index in low_level_indices], axis=0)
		mean_low_u = np.mean(selected_u, axis=0)
		mean_low_v = np.mean(selected_v, axis=0)
		low_level_mean_wind_speed = np.sqrt(mean_low_u * mean_low_u + mean_low_v * mean_low_v + epsilon).astype(np.float32, copy=False)
		feature_groups.append(low_level_mean_wind_speed[:, :, None].astype(np.float32, copy=False))

	if atmospheric["add_updraft"] and updraft_levels:
		feature_groups.extend(updraft_levels)

	if atmospheric["add_wind_direction"]:
		wind_direction_features: list[np.ndarray] = []
		for z_level in range(num_vertical_levels):
			u_values = u_levels[z_level]
			v_values = v_levels[z_level]
			wind_speed = np.sqrt(u_values * u_values + v_values * v_values + epsilon).astype(np.float32, copy=False)
			wind_dir_cos = (u_values / wind_speed).astype(np.float32, copy=False)
			wind_dir_sin = (v_values / wind_speed).astype(np.float32, copy=False)
			wind_direction_features.append(wind_dir_cos[:, :, None].astype(np.float32, copy=False))
			wind_direction_features.append(wind_dir_sin[:, :, None].astype(np.float32, copy=False))
		feature_groups.extend(wind_direction_features)

	if not feature_groups:
		height, width = int(raw_frame.shape[0]), int(raw_frame.shape[1])
		return np.zeros((height, width, 0), dtype=np.float32)
	return np.concatenate(feature_groups, axis=-1).astype(np.float32, copy=False)


def resolve_engineered_feature_slices(config: Mapping[str, Any], base_input_channel_count: int) -> dict[str, slice]:
	"""Resolve deterministic channel slices for all engineered feature groups."""

	offset = int(base_input_channel_count)
	atmospheric = _resolve_atmospheric_features_config(config)
	engineered = _resolve_engineered_features_config(config)
	energy_release = resolve_energy_release_config(config)
	layout = _resolve_channel_layout(config)
	slices: dict[str, slice] = {}

	if atmospheric["enabled"] and atmospheric["add_horizontal_wind_speed"]:
		num_levels = int(atmospheric["num_vertical_levels"])
		slices["horizontal_wind_speed"] = slice(offset, offset + num_levels)
		offset += num_levels
	if atmospheric["enabled"] and atmospheric["add_low_level_mean_wind_speed"]:
		slices["low_level_mean_wind_speed"] = slice(offset, offset + 1)
		offset += 1
	if atmospheric["enabled"] and atmospheric["add_updraft"]:
		num_levels = int(atmospheric["num_vertical_levels"])
		slices["updraft"] = slice(offset, offset + num_levels)
		offset += num_levels
	if atmospheric["enabled"] and atmospheric["add_wind_direction"]:
		num_levels = int(atmospheric["num_vertical_levels"])
		slices["wind_direction"] = slice(offset, offset + 2 * num_levels)
		slices["wind_dir_cos"] = slice(offset, offset + 2 * num_levels, 2)
		slices["wind_dir_sin"] = slice(offset + 1, offset + 2 * num_levels, 2)
		offset += 2 * num_levels
	if engineered["enabled"] and engineered["add_flux_delta"]:
		slices["flux_delta"] = slice(offset, offset + len(layout["flux_channels"]))
		offset += len(layout["flux_channels"])
	if engineered["enabled"] and engineered["add_fuel_delta"]:
		slices["fuel_delta"] = slice(offset, offset + len(layout["fuel_channels"]))
		offset += len(layout["fuel_channels"])
	if engineered["enabled"] and engineered["add_step_consumed_fuel"]:
		slices["step_consumed_fuel"] = slice(offset, offset + len(layout["fuel_channels"]))
		offset += len(layout["fuel_channels"])
	if engineered["enabled"] and engineered["add_cumulative_consumed_fuel"]:
		slices["cumulative_consumed_fuel"] = slice(offset, offset + len(layout["fuel_channels"]))
		offset += len(layout["fuel_channels"])
	if energy_release["enabled"] and energy_release["add_as_input_history"]:
		slices["energy_release_history"] = slice(offset, offset + 1)
		offset += 1
	return slices


def _load_initial_fuel_map(file_paths: Sequence[Path], config: Mapping[str, Any], initial_index: int = 0) -> np.ndarray:
	"""Load the initial surface/canopy fuel map used for cumulative consumed-fuel features."""

	engineered = _resolve_engineered_features_config(config)
	if engineered["initial_fuel_mode"] != "first_dataset_frame":
		raise ValueError(
			f"Unsupported engineered_features.initial_fuel_mode: {engineered['initial_fuel_mode']!r}. "
			"Only 'first_dataset_frame' is currently supported."
		)
	layout = _resolve_channel_layout(config)
	initial_index = max(0, min(int(initial_index), len(file_paths) - 1))
	first_frame = np.load(file_paths[initial_index], mmap_mode="r", allow_pickle=False)
	return _slice_channels(first_frame, layout["fuel_channels"])


def build_engineered_features(
	input_frames: np.ndarray,
	file_paths: Sequence[str | Path],
	start_index: int,
	config: Mapping[str, Any],
	energy_geometry: Mapping[str, Any] | None = None,
	initial_fuel_index: int = 0,
) -> np.ndarray:
	"""Build leakage-safe engineered features from current and previous frames only."""

	engineered = _resolve_engineered_features_config(config)
	energy_release = resolve_energy_release_config(config)
	atmospheric_count = count_atmospheric_engineered_channels(config)
	fuel_flux_count = _count_fuel_flux_engineered_channels(config)
	energy_history_count = 1 if energy_release["enabled"] and energy_release["add_as_input_history"] else 0
	if atmospheric_count + fuel_flux_count + energy_history_count <= 0:
		height, width = int(input_frames.shape[1]), int(input_frames.shape[2])
		return np.zeros((int(input_frames.shape[0]), height, width, 0), dtype=np.float32)

	layout = _resolve_channel_layout(config)
	resolved_paths = [Path(path) for path in file_paths]
	raw_frames = np.asarray(input_frames, dtype=np.float32)
	if raw_frames.ndim != 4:
		raise ValueError(f"build_engineered_features expects raw input_frames shaped (T, H, W, C), got {raw_frames.shape}.")

	initial_fuel = None
	if engineered["enabled"] and engineered["add_cumulative_consumed_fuel"]:
		initial_fuel = _load_initial_fuel_map(resolved_paths, config, initial_index=initial_fuel_index)
	feature_frames: list[np.ndarray] = []
	for timestep_index in range(raw_frames.shape[0]):
		global_frame_index = int(start_index) + timestep_index
		current_frame = raw_frames[timestep_index]

		per_timestep_features: list[np.ndarray] = []
		atmospheric_features = build_atmospheric_features(current_frame, config)
		if atmospheric_features.shape[-1] != atmospheric_count:
			raise ValueError(
				"Atmospheric engineered feature count mismatch. "
				f"Expected {atmospheric_count}, got {atmospheric_features.shape[-1]}."
			)
		if atmospheric_features.shape[-1] > 0:
			per_timestep_features.append(atmospheric_features)

		if engineered["enabled"]:
			if global_frame_index > 0:
				previous_frame = np.load(resolved_paths[global_frame_index - 1], mmap_mode="r", allow_pickle=False)
			else:
				previous_frame = current_frame

			current_flux = _slice_channels(current_frame, layout["flux_channels"])
			previous_flux = _slice_channels(previous_frame, layout["flux_channels"])
			current_fuel = _slice_channels(current_frame, layout["fuel_channels"])
			previous_fuel = _slice_channels(previous_frame, layout["fuel_channels"])

			if engineered["add_flux_delta"]:
				per_timestep_features.append(current_flux - previous_flux)
			if engineered["add_fuel_delta"]:
				per_timestep_features.append(current_fuel - previous_fuel)
			if engineered["add_step_consumed_fuel"]:
				step_consumed = previous_fuel - current_fuel
				if engineered["clamp_consumed_fuel_nonnegative"]:
					step_consumed = np.maximum(step_consumed, 0.0)
				per_timestep_features.append(step_consumed)
			if engineered["add_cumulative_consumed_fuel"]:
				if initial_fuel is None:
					raise ValueError("initial_fuel must be available when add_cumulative_consumed_fuel is enabled.")
				cumulative_consumed = initial_fuel - current_fuel
				if engineered["clamp_consumed_fuel_nonnegative"]:
					cumulative_consumed = np.maximum(cumulative_consumed, 0.0)
				per_timestep_features.append(cumulative_consumed)
		if energy_history_count:
			if energy_geometry is None:
				raise ValueError("energy_geometry is required when energy_release.add_as_input_history=true.")
			energy_maps = compute_energy_release_maps(
				current_frame,
				config=config,
				area_2d_m2=np.asarray(energy_geometry["area_2d_m2"], dtype=np.float32),
			)
			energy_history = transform_energy_target(energy_maps["energy_release_total_MW"], config)
			per_timestep_features.append(energy_history[:, :, None].astype(np.float32, copy=False))

		if per_timestep_features:
			feature_frames.append(np.concatenate(per_timestep_features, axis=-1).astype(np.float32, copy=False))
		else:
			height, width = current_frame.shape[:2]
			feature_frames.append(np.zeros((height, width, 0), dtype=np.float32))

	return np.stack(feature_frames, axis=0).astype(np.float32, copy=False)


def build_engineered_features_from_raw_window(
	raw_window: np.ndarray,
	config: Mapping[str, Any],
	initial_fuel: np.ndarray | None = None,
	previous_frame_before_window: np.ndarray | None = None,
	energy_geometry: Mapping[str, Any] | None = None,
) -> np.ndarray:
	"""Build engineered features from an in-memory rollout window.

	This mirrors ``build_engineered_features`` but avoids disk-based lookups so
	autoregressive rollout can rebuild deltas and consumed-fuel features from the
	current predicted raw window at each step.
	"""

	raw_frames = np.asarray(raw_window, dtype=np.float32)
	if raw_frames.ndim != 4:
		raise ValueError(
			"build_engineered_features_from_raw_window expects raw_window shaped (T, H, W, C), "
			f"got {raw_frames.shape}."
		)

	engineered = _resolve_engineered_features_config(config)
	energy_release = resolve_energy_release_config(config)
	atmospheric_count = count_atmospheric_engineered_channels(config)
	fuel_flux_count = _count_fuel_flux_engineered_channels(config)
	energy_history_count = 1 if energy_release["enabled"] and energy_release["add_as_input_history"] else 0
	if atmospheric_count + fuel_flux_count + energy_history_count <= 0:
		height, width = int(raw_frames.shape[1]), int(raw_frames.shape[2])
		return np.zeros((int(raw_frames.shape[0]), height, width, 0), dtype=np.float32)

	layout = _resolve_channel_layout(config)
	if engineered["enabled"] and engineered["add_cumulative_consumed_fuel"] and initial_fuel is None:
		raise ValueError(
			"initial_fuel must be provided when add_cumulative_consumed_fuel is enabled for rollout preprocessing."
		)

	if previous_frame_before_window is not None:
		previous_frame_before_window = np.asarray(previous_frame_before_window, dtype=np.float32)
		if previous_frame_before_window.shape != raw_frames[0].shape:
			raise ValueError(
				"previous_frame_before_window must match one raw frame shape. "
				f"Expected {raw_frames[0].shape}, got {previous_frame_before_window.shape}."
			)

	feature_frames: list[np.ndarray] = []
	for timestep_index in range(raw_frames.shape[0]):
		current_frame = raw_frames[timestep_index]
		if timestep_index == 0:
			previous_frame = previous_frame_before_window if previous_frame_before_window is not None else current_frame
		else:
			previous_frame = raw_frames[timestep_index - 1]

		per_timestep_features: list[np.ndarray] = []
		atmospheric_features = build_atmospheric_features(current_frame, config)
		if atmospheric_features.shape[-1] != atmospheric_count:
			raise ValueError(
				"Atmospheric engineered feature count mismatch in rollout preprocessing. "
				f"Expected {atmospheric_count}, got {atmospheric_features.shape[-1]}."
			)
		if atmospheric_features.shape[-1] > 0:
			per_timestep_features.append(atmospheric_features)

		if engineered["enabled"]:
			current_flux = _slice_channels(current_frame, layout["flux_channels"])
			previous_flux = _slice_channels(previous_frame, layout["flux_channels"])
			current_fuel = _slice_channels(current_frame, layout["fuel_channels"])
			previous_fuel = _slice_channels(previous_frame, layout["fuel_channels"])

			if engineered["add_flux_delta"]:
				per_timestep_features.append(current_flux - previous_flux)
			if engineered["add_fuel_delta"]:
				per_timestep_features.append(current_fuel - previous_fuel)
			if engineered["add_step_consumed_fuel"]:
				step_consumed = previous_fuel - current_fuel
				if engineered["clamp_consumed_fuel_nonnegative"]:
					step_consumed = np.maximum(step_consumed, 0.0)
				per_timestep_features.append(step_consumed)
			if engineered["add_cumulative_consumed_fuel"]:
				cumulative_consumed = np.asarray(initial_fuel, dtype=np.float32) - current_fuel
				if engineered["clamp_consumed_fuel_nonnegative"]:
					cumulative_consumed = np.maximum(cumulative_consumed, 0.0)
				per_timestep_features.append(cumulative_consumed)
		if energy_history_count:
			if energy_geometry is None:
				raise ValueError("energy_geometry is required when energy_release.add_as_input_history=true.")
			energy_maps = compute_energy_release_maps(
				current_frame,
				config=config,
				area_2d_m2=np.asarray(energy_geometry["area_2d_m2"], dtype=np.float32),
			)
			energy_history = transform_energy_target(energy_maps["energy_release_total_MW"], config)
			per_timestep_features.append(energy_history[:, :, None].astype(np.float32, copy=False))

		if per_timestep_features:
			feature_frames.append(np.concatenate(per_timestep_features, axis=-1).astype(np.float32, copy=False))
		else:
			height, width = current_frame.shape[:2]
			feature_frames.append(np.zeros((height, width, 0), dtype=np.float32))

	return np.stack(feature_frames, axis=0).astype(np.float32, copy=False)


def build_model_input_from_raw_window(
	raw_window: np.ndarray,
	config: Mapping[str, Any],
	normalization_stats: Mapping[str, np.ndarray] | None = None,
	initial_fuel: np.ndarray | None = None,
	previous_frame_before_window: np.ndarray | None = None,
	energy_geometry: Mapping[str, Any] | None = None,
) -> np.ndarray:
	"""Convert a raw rollout window into the model's normalized (1, T, C, H, W) input."""

	raw_frames = np.asarray(raw_window, dtype=np.float32)
	if raw_frames.ndim != 4:
		raise ValueError(
			"build_model_input_from_raw_window expects raw_window shaped (T, H, W, C), "
			f"got {raw_frames.shape}."
		)

	input_channel_count = int(config.get("input_channel_count", raw_frames.shape[-1]))
	base_input_channel_indices = _resolve_input_channel_indices(config, input_channel_count)
	base_input_frames = [_slice_channels(raw_frame, base_input_channel_indices) for raw_frame in raw_frames]
	stacked_inputs = np.stack(base_input_frames, axis=0).astype(np.float32, copy=False)

	engineered_inputs = build_engineered_features_from_raw_window(
		raw_window=raw_frames,
		config=config,
		initial_fuel=initial_fuel,
		previous_frame_before_window=previous_frame_before_window,
		energy_geometry=energy_geometry,
	)
	if engineered_inputs.shape[:3] != stacked_inputs.shape[:3]:
		raise ValueError(
			"Engineered rollout features must align with base inputs in (T, H, W). "
			f"Got base={stacked_inputs.shape} engineered={engineered_inputs.shape}."
		)
	if engineered_inputs.shape[-1] > 0:
		stacked_inputs = np.concatenate([stacked_inputs, engineered_inputs], axis=-1)

	total_input_channels = stacked_inputs.shape[-1]
	model_input_channels = int(config.get("model", {}).get("input_channels", total_input_channels))
	if model_input_channels != total_input_channels:
		raise ValueError(
			"Model input channel count does not match rollout-preprocessed channels. "
			f"model.input_channels={model_input_channels}, actual={total_input_channels}."
		)

	if normalization_stats is not None:
		stats_mean = np.asarray(normalization_stats["mean"], dtype=np.float32)
		stats_std = np.asarray(normalization_stats["std"], dtype=np.float32)
		if stats_mean.shape[0] != total_input_channels or stats_std.shape[0] != total_input_channels:
			raise ValueError(
				"Normalization stats channel count does not match rollout-preprocessed inputs. "
				f"Need {total_input_channels}, got mean={stats_mean.shape[0]} std={stats_std.shape[0]}."
			)
		stacked_inputs = normalize_tensor(stacked_inputs, stats_mean, stats_std).astype(np.float32, copy=False)

	stacked_inputs = np.transpose(stacked_inputs, (0, 3, 1, 2))
	stacked_inputs = np.ascontiguousarray(stacked_inputs, dtype=np.float32)
	return np.expand_dims(stacked_inputs, axis=0)


def build_multitask_target(
	current_frame: np.ndarray,
	future_frame: np.ndarray,
	initial_fuel: np.ndarray,
	config: Mapping[str, Any],
	energy_geometry: Mapping[str, Any] | None = None,
) -> np.ndarray:
	"""Build the multitask target, optionally including energy release channels."""

	multitask = _resolve_multitask_config(config)
	surface_fuel_channel = int(multitask["surface_fuel_channel"])
	canopy_fuel_channel = int(multitask["canopy_fuel_channel"])
	flux_mask_channel = int(multitask["flux_mask_channel"])

	current_surface_fuel = np.asarray(current_frame[:, :, surface_fuel_channel], dtype=np.float32)
	future_surface_fuel = np.asarray(future_frame[:, :, surface_fuel_channel], dtype=np.float32)
	current_canopy_fuel = np.asarray(current_frame[:, :, canopy_fuel_channel], dtype=np.float32)
	future_canopy_fuel = np.asarray(future_frame[:, :, canopy_fuel_channel], dtype=np.float32)

	surface_consumed_target = current_surface_fuel - future_surface_fuel
	canopy_consumed_target = current_canopy_fuel - future_canopy_fuel
	if multitask["clamp_consumed_fuel_targets_nonnegative"]:
		surface_consumed_target = np.maximum(surface_consumed_target, 0.0)
		canopy_consumed_target = np.maximum(canopy_consumed_target, 0.0)

	mask_target_type = str(multitask["mask_target_type"]).lower()
	if mask_target_type == "active_flux":
		future_flux = np.asarray(future_frame[:, :, flux_mask_channel], dtype=np.float32)
		mask = future_flux > float(multitask["flux_fire_threshold"])
	elif mask_target_type == "burned_fuel":
		initial_surface_fuel = np.asarray(initial_fuel[:, :, 0], dtype=np.float32)
		initial_canopy_fuel = np.asarray(initial_fuel[:, :, 1], dtype=np.float32)
		surface_cumulative_consumed = initial_surface_fuel - future_surface_fuel
		canopy_cumulative_consumed = initial_canopy_fuel - future_canopy_fuel
		if multitask["clamp_consumed_fuel_targets_nonnegative"]:
			surface_cumulative_consumed = np.maximum(surface_cumulative_consumed, 0.0)
			canopy_cumulative_consumed = np.maximum(canopy_cumulative_consumed, 0.0)
		combined_cumulative_consumed = np.maximum(surface_cumulative_consumed, canopy_cumulative_consumed)
		mask = combined_cumulative_consumed > float(multitask["consumed_fuel_threshold"])
	else:
		raise ValueError(
			"Unsupported multitask.mask_target_type. "
			f"Expected 'active_flux' or 'burned_fuel', got {mask_target_type!r}."
		)

	mask_array = np.asarray(mask, dtype=np.float32)
	if not np.all(np.isin(np.unique(mask_array), np.asarray([0.0, 1.0], dtype=np.float32))):
		raise ValueError("Multitask mask target must contain only 0.0 and 1.0 values.")
	if not np.isfinite(surface_consumed_target).all() or not np.isfinite(canopy_consumed_target).all():
		raise ValueError("Multitask regression targets contain non-finite values.")

	target_channels = [
		surface_consumed_target.astype(np.float32, copy=False),
		canopy_consumed_target.astype(np.float32, copy=False),
		mask_array,
	]
	energy_release = resolve_energy_release_config(config)
	energy_output_names = resolve_energy_output_channel_names(config)
	if energy_release["enabled"] and energy_output_names:
		if energy_geometry is None:
			raise ValueError("energy_geometry must be provided when energy_release.enabled=true.")
		energy_maps = compute_energy_release_maps(
			future_frame,
			config=config,
			area_2d_m2=np.asarray(energy_geometry["area_2d_m2"], dtype=np.float32),
		)
		for energy_output_name in energy_output_names:
			target_channels.append(transform_energy_target(energy_maps[energy_output_name], config))

	target = np.stack(target_channels, axis=0)
	return np.ascontiguousarray(target, dtype=np.float32)


class FireSequenceDataset(Dataset):
	"""Dataset for forecasting a target map from a sequence of preceding maps."""

	def __init__(
		self,
		file_paths: Iterable[str | Path],
		sample_indices: Sequence[int] | None,
		input_sequence_length: int,
		prediction_horizon: int,
		target_channel: int,
		input_channel_count: int | None = None,
		input_channel_indices: Sequence[int] | None = None,
		task_type: str = "regression",
		fire_threshold: float = 0.5,
		use_patches: bool = False,
		patch_size: int = 64,
		active_patch_probability: float = 0.7,
		active_threshold: float = 0.0,
		normalization_stats: Mapping[str, np.ndarray] | str | Path | None = None,
		normalize_target: bool = False,
		transform=None,
		target_transform=None,
		return_metadata: bool = False,
		config: Mapping[str, Any] | None = None,
	) -> None:
		self.file_paths = _sort_chronologically(_as_path_list(file_paths))
		self.input_sequence_length = int(input_sequence_length)
		self.prediction_horizon = int(prediction_horizon)
		self.target_channel = int(target_channel)
		self.input_channel_count = None if input_channel_count is None else int(input_channel_count)
		self.input_channel_indices = _coerce_index_list(input_channel_indices)
		self.task_type = str(task_type).lower()
		self.fire_threshold = float(fire_threshold)
		self.use_patches = bool(use_patches)
		self.patch_size = int(patch_size)
		self.active_patch_probability = float(active_patch_probability)
		self.active_threshold = float(active_threshold)
		self.transform = transform
		self.target_transform = target_transform
		self.return_metadata = bool(return_metadata)
		self.config = dict(config) if isinstance(config, Mapping) else {
			"task_type": self.task_type,
			"target_channel": self.target_channel,
			"input_channel_count": self.input_channel_count,
			"input_channel_indices": self.input_channel_indices,
			"fire_threshold": self.fire_threshold,
			"active_threshold": self.active_threshold,
		}

		target_normalization_config = _resolve_target_normalization_config(self.config)
		self.normalize_target = bool(normalize_target or target_normalization_config["enabled"])

		if self.input_sequence_length <= 0:
			raise ValueError(f"input_sequence_length must be positive, got {self.input_sequence_length}.")
		if self.prediction_horizon < 0:
			raise ValueError(f"prediction_horizon must be non-negative, got {self.prediction_horizon}.")
		if not self.file_paths:
			raise ValueError("FireSequenceDataset requires at least one file path.")

		missing_files = [str(path) for path in self.file_paths if not path.exists()]
		if missing_files:
			raise FileNotFoundError(
				"The following dataset files do not exist:\n" + "\n".join(f"  {path}" for path in missing_files)
			)

		first_tensor = self._load_tensor(self.file_paths[0])
		if first_tensor.ndim != 3:
			raise ValueError(
				f"Expected dataset files to contain 3D tensors, got shape {first_tensor.shape} in {self.file_paths[0]}."
			)
		self.expected_height, self.expected_width, self.num_channels = first_tensor.shape

		if self.input_channel_count is None:
			self.input_channel_count = self.num_channels
		if self.input_channel_count <= 0 or self.input_channel_count > self.num_channels:
			raise ValueError(f"input_channel_count must be in [1, {self.num_channels}], got {self.input_channel_count}.")
		if self.target_channel < 0 or self.target_channel >= self.num_channels:
			raise ValueError(f"target_channel must be in [0, {self.num_channels - 1}], got {self.target_channel}.")
		if self.task_type not in {"regression", "segmentation", "multitask"}:
			raise ValueError(
				f"task_type must be 'regression', 'segmentation', or 'multitask', got {self.task_type!r}."
			)
		if not 0.0 <= self.active_patch_probability <= 1.0:
			raise ValueError("active_patch_probability must be in [0, 1], got {self.active_patch_probability}.")
		if self.patch_size <= 0:
			raise ValueError(f"patch_size must be positive, got {self.patch_size}.")
		if self.use_patches and (self.patch_size > self.expected_height or self.patch_size > self.expected_width):
			raise ValueError(
				"patch_size must be <= both spatial dimensions. "
				f"Got patch_size={self.patch_size}, H={self.expected_height}, W={self.expected_width}."
			)

		self.base_input_channel_indices = _resolve_input_channel_indices(self.config, self.input_channel_count)
		if any(index < 0 or index >= self.num_channels for index in self.base_input_channel_indices):
			raise ValueError(
				f"input_channel_indices must stay within [0, {self.num_channels - 1}], got {self.base_input_channel_indices}."
			)
		self.base_input_channel_count = len(self.base_input_channel_indices)
		self.fuel_flux_engineered_channel_count = _count_fuel_flux_engineered_channels(self.config)
		self.atmospheric_engineered_channel_count = count_atmospheric_engineered_channels(self.config)
		self.energy_release_config = resolve_energy_release_config(self.config)
		self.energy_target_channel_count = resolve_energy_target_count(self.config)
		self.energy_history_channel_count = 1 if self.energy_release_config["enabled"] and self.energy_release_config["add_as_input_history"] else 0
		self.engineered_channel_count = (
			self.fuel_flux_engineered_channel_count
			+ self.atmospheric_engineered_channel_count
			+ self.energy_history_channel_count
		)
		self.total_input_channels = self.base_input_channel_count + self.engineered_channel_count
		self.input_channels_after_engineering = self.total_input_channels
		self.engineered_feature_slices = resolve_engineered_feature_slices(self.config, self.base_input_channel_count)

		max_valid_start = len(self.file_paths) - self.input_sequence_length - self.prediction_horizon
		if max_valid_start < 0:
			raise ValueError(
				"Not enough files to form even one sample. "
				f"Need at least input_sequence_length + prediction_horizon = "
				f"{self.input_sequence_length + self.prediction_horizon}, got {len(self.file_paths)}."
			)

		if sample_indices is None:
			self.sample_indices = list(range(max_valid_start + 1))
		else:
			self.sample_indices = [int(index) for index in sample_indices]
		invalid_indices = [index for index in self.sample_indices if index < 0 or index > max_valid_start]
		if invalid_indices:
			raise ValueError(
				"sample_indices contain invalid sample start positions. "
				f"Valid range is [0, {max_valid_start}], got {invalid_indices[:10]}."
			)

		self.normalization_stats = self._coerce_normalization_stats(normalization_stats)
		self.input_normalization_on_device = bool(
			self.normalization_stats is not None and input_normalization_runs_on_device(self.config)
		)
		self.inputs_are_normalized = bool(
			self.normalization_stats is not None and not self.input_normalization_on_device
		)
		self.target_mean, self.target_std = self._resolve_target_normalization_stats()
		self.initial_fuel_map = _load_initial_fuel_map(self.file_paths, self.config) if self.task_type == "multitask" or _resolve_engineered_features_config(self.config)["enabled"] else None
		self.energy_geometry = None
		if self.energy_release_config["enabled"] and (self.task_type == "multitask" or self.energy_history_channel_count > 0):
			dataset_dir = self.file_paths[0].parent
			self.energy_geometry = load_fire_geometry(
				data_dir=dataset_dir,
				config=self.config,
				expected_shape=(self.expected_height, self.expected_width),
			)
			print(
				"Energy release geometry | "
				f"dataset={dataset_dir.name} geom={self.energy_geometry['geom_path']} "
				f"area_min_m2={self.energy_geometry['area_min_m2']:.6f} "
				f"area_mean_m2={self.energy_geometry['area_mean_m2']:.6f} "
				f"area_max_m2={self.energy_geometry['area_max_m2']:.6f}"
			)

	def _coerce_normalization_stats(
		self,
		normalization_stats: Mapping[str, np.ndarray] | str | Path | None,
	) -> dict[str, np.ndarray] | None:
		"""Normalize the normalization-statistics input into a usable dictionary."""

		if normalization_stats is None:
			return None
		if isinstance(normalization_stats, (str, Path)):
			stats = load_normalization_stats(normalization_stats)
		else:
			stats = dict(normalization_stats)

		required_keys = {"mean", "std", "min", "max"}
		missing = required_keys.difference(stats)
		if missing:
			raise KeyError(f"Normalization stats are missing required key(s): {', '.join(sorted(missing))}")

		normalized_stats = {key: np.asarray(stats[key]) for key in required_keys}
		for optional_key in (
			"target_mean",
			"target_std",
			"target_min",
			"target_max",
			"multitask_target_mean",
			"multitask_target_std",
		):
			if optional_key in stats:
				normalized_stats[optional_key] = np.asarray(stats[optional_key])
		return normalized_stats

	def _resolve_target_normalization_stats(self) -> tuple[float | np.ndarray | None, float | np.ndarray | None]:
		"""Resolve target stats for optional target normalization."""

		if self.normalization_stats is None or not self.normalize_target:
			return None, None

		if self.task_type == "regression":
			stats_mean = np.asarray(self.normalization_stats["mean"])
			stats_std = np.asarray(self.normalization_stats["std"])
			if self.target_channel < stats_mean.shape[0] and self.target_channel < stats_std.shape[0]:
				return float(stats_mean[self.target_channel]), float(stats_std[self.target_channel])
			if "target_mean" in self.normalization_stats and "target_std" in self.normalization_stats:
				return (
					float(np.asarray(self.normalization_stats["target_mean"])),
					float(np.asarray(self.normalization_stats["target_std"])),
				)
			raise ValueError(
				"Target normalization was requested, but normalization stats do not include "
				f"target channel {self.target_channel}."
			)

		if self.task_type == "multitask":
			if "multitask_target_mean" in self.normalization_stats and "multitask_target_std" in self.normalization_stats:
				mean = np.asarray(self.normalization_stats["multitask_target_mean"], dtype=np.float32)
				std = np.asarray(self.normalization_stats["multitask_target_std"], dtype=np.float32)
				if mean.shape[0] < 2 or std.shape[0] < 2:
					raise ValueError("multitask target normalization stats must contain at least two channels.")
				return mean[:2], std[:2]
			return None, None

		return None, None

	def _load_tensor(self, file_path: Path) -> np.ndarray:
		"""Load a single tensor from disk with validation-friendly settings."""

		return np.load(file_path, mmap_mode="r", allow_pickle=False)

	def _validate_tensor_shape(self, tensor: np.ndarray, file_path: Path) -> None:
		"""Ensure a loaded tensor matches the expected spatial and channel dimensions."""

		if tensor.ndim != 3:
			raise ValueError(f"Expected a 3D tensor in {file_path}, got shape {tensor.shape}.")
		if tensor.shape != (self.expected_height, self.expected_width, self.num_channels):
			raise ValueError(
				f"Inconsistent tensor shape in {file_path}. "
				f"Expected {(self.expected_height, self.expected_width, self.num_channels)}, got {tensor.shape}."
			)

	def _sample_patch_origin(self, target_map_for_sampling: np.ndarray) -> tuple[int, int]:
		"""Choose the top-left patch origin, preferring active fire areas when requested."""

		height, width = target_map_for_sampling.shape
		max_top = height - self.patch_size
		max_left = width - self.patch_size
		use_active_patch = np.random.random() < self.active_patch_probability
		if use_active_patch:
			active_pixels = np.argwhere(target_map_for_sampling > self.active_threshold)
			if active_pixels.size > 0:
				center_y, center_x = active_pixels[np.random.randint(active_pixels.shape[0])]
				top = int(center_y) - self.patch_size // 2
				left = int(center_x) - self.patch_size // 2
				top = max(0, min(top, max_top))
				left = max(0, min(left, max_left))
				return top, left

		top = int(np.random.randint(0, max_top + 1)) if max_top > 0 else 0
		left = int(np.random.randint(0, max_left + 1)) if max_left > 0 else 0
		return top, left

	def _normalize_inputs(self, stacked_inputs: np.ndarray) -> np.ndarray:
		"""Normalize input channels using post-engineering statistics."""

		if self.normalization_stats is None or self.input_normalization_on_device:
			return stacked_inputs

		stats_mean = np.asarray(self.normalization_stats["mean"], dtype=np.float32)
		stats_std = np.asarray(self.normalization_stats["std"], dtype=np.float32)
		channel_count = stacked_inputs.shape[-1]
		if stats_mean.shape[0] != channel_count or stats_std.shape[0] != channel_count:
			if stats_mean.shape[0] < channel_count or stats_std.shape[0] < channel_count:
				raise ValueError(
					"Normalization stats channel count does not match engineered inputs. "
					f"Need {channel_count}, got mean={stats_mean.shape[0]} std={stats_std.shape[0]}."
				)
			stats_mean = stats_mean[:channel_count]
			stats_std = stats_std[:channel_count]
		return normalize_tensor(stacked_inputs, stats_mean, stats_std).astype(np.float32, copy=False)

	def _normalize_target(self, target_array: np.ndarray) -> np.ndarray:
		"""Normalize target channels when configured."""

		if self.target_mean is None or self.target_std is None:
			return target_array

		if self.task_type == "regression":
			return normalize_channel_map(target_array, self.target_mean, self.target_std).astype(np.float32, copy=False)

		if self.task_type == "multitask":
			mean = np.asarray(self.target_mean, dtype=np.float32)
			std = np.asarray(self.target_std, dtype=np.float32)
			if mean.shape[0] < 2 or std.shape[0] < 2:
				raise ValueError("Multitask target normalization stats must provide at least two channels.")
			target_array = np.asarray(target_array, dtype=np.float32).copy()
			target_array[0] = normalize_channel_map(target_array[0], mean[0], std[0])
			target_array[1] = normalize_channel_map(target_array[1], mean[1], std[1])
			return target_array.astype(np.float32, copy=False)

		return target_array

	def __len__(self) -> int:
		return len(self.sample_indices)

	def __getitem__(self, index: int):
		if torch is None:
			raise ImportError("PyTorch is required to index FireSequenceDataset.")

		sample_start = self.sample_indices[index]
		current_index = sample_start + self.input_sequence_length - 1
		future_index = current_index + self.prediction_horizon
		input_file_paths = self.file_paths[sample_start : sample_start + self.input_sequence_length]
		current_file_path = self.file_paths[current_index]
		target_file_path = self.file_paths[future_index]

		raw_input_frames: list[np.ndarray] = []
		base_input_frames: list[np.ndarray] = []
		for file_path in input_file_paths:
			tensor = self._load_tensor(file_path)
			self._validate_tensor_shape(tensor, file_path)
			raw_frame = np.asarray(tensor, dtype=np.float32)
			raw_input_frames.append(raw_frame)
			base_input_frames.append(_slice_channels(raw_frame, self.base_input_channel_indices))

		current_tensor = raw_input_frames[-1]
		future_tensor = self._load_tensor(target_file_path)
		self._validate_tensor_shape(future_tensor, target_file_path)
		future_tensor = np.asarray(future_tensor, dtype=np.float32)

		if self.task_type == "multitask":
			if self.initial_fuel_map is None:
				raise ValueError("initial_fuel_map must be available for multitask targets.")
			target_array = build_multitask_target(
				current_frame=current_tensor,
				future_frame=future_tensor,
				initial_fuel=self.initial_fuel_map,
				config=self.config,
				energy_geometry=self.energy_geometry,
			)
			target_map_for_sampling = np.asarray(target_array[2], dtype=np.float32)
		else:
			raw_target_array = np.asarray(future_tensor[:, :, self.target_channel], dtype=np.float32)
			target_array = raw_target_array.copy()
			if self.task_type == "segmentation":
				target_array = (target_array > self.fire_threshold).astype(np.float32, copy=False)
			target_map_for_sampling = raw_target_array

		stacked_inputs = np.stack(base_input_frames, axis=0).astype(np.float32, copy=False)
		engineered_inputs = build_engineered_features(
			input_frames=np.stack(raw_input_frames, axis=0).astype(np.float32, copy=False),
			file_paths=self.file_paths,
			start_index=sample_start,
			config=self.config,
			energy_geometry=self.energy_geometry,
		)
		if engineered_inputs.shape[:3] != stacked_inputs.shape[:3]:
			raise ValueError(
				"Engineered feature tensor must align with base inputs in (T, H, W). "
				f"Got base={stacked_inputs.shape} engineered={engineered_inputs.shape}."
			)
		if engineered_inputs.shape[-1] != self.engineered_channel_count:
			raise ValueError(
				f"Expected {self.engineered_channel_count} engineered channels, got {engineered_inputs.shape[-1]}."
			)
		if engineered_inputs.shape[-1] > 0:
			stacked_inputs = np.concatenate([stacked_inputs, engineered_inputs], axis=-1)

		patch_top = None
		patch_left = None
		if self.use_patches:
			patch_top, patch_left = self._sample_patch_origin(target_map_for_sampling)
			patch_bottom = patch_top + self.patch_size
			patch_right = patch_left + self.patch_size
			stacked_inputs = stacked_inputs[:, patch_top:patch_bottom, patch_left:patch_right, :]
			if self.task_type == "multitask":
				target_array = target_array[:, patch_top:patch_bottom, patch_left:patch_right]
			else:
				target_array = target_array[patch_top:patch_bottom, patch_left:patch_right]

		stacked_inputs = self._normalize_inputs(stacked_inputs)
		target_array = self._normalize_target(target_array)

		stacked_inputs = np.transpose(stacked_inputs, (0, 3, 1, 2))
		stacked_inputs = np.ascontiguousarray(stacked_inputs, dtype=np.float32)
		if self.task_type == "multitask":
			target_array = np.ascontiguousarray(target_array, dtype=np.float32)
			if not np.all(np.isin(np.unique(target_array[2]), np.asarray([0.0, 1.0], dtype=np.float32))):
				raise ValueError("Multitask mask channel must contain only 0.0 and 1.0 after processing.")
		else:
			target_array = np.expand_dims(np.ascontiguousarray(target_array, dtype=np.float32), axis=0)

		if stacked_inputs.shape[1] != self.total_input_channels:
			raise ValueError(
				f"Expected stacked input channel dimension {self.total_input_channels}, got {stacked_inputs.shape[1]}."
			)
		if self.task_type == "multitask" and target_array.shape[0] != _multitask_output_channel_count(self.config):
			raise ValueError(
				f"Expected multitask target shape ({_multitask_output_channel_count(self.config)}, H, W), got {target_array.shape}."
			)

		x_tensor = torch.from_numpy(stacked_inputs).to(torch.float32)
		y_tensor = torch.from_numpy(target_array).to(torch.float32)
		if self.transform is not None:
			x_tensor = self.transform(x_tensor)
		if self.target_transform is not None:
			y_tensor = self.target_transform(y_tensor)

		if self.return_metadata:
			input_indices = list(range(sample_start, sample_start + self.input_sequence_length))
			target_offsets = temporal_target_offsets(
				{
					"input_sequence_length": self.input_sequence_length,
					"prediction_horizon": self.prediction_horizon,
				}
			)
			metadata = {
				"sample_index": sample_start,
				"start_idx": sample_start,
				"input_indices": input_indices,
				"last_input_idx": current_index,
				"target_idx": future_index,
				"current_idx": current_index,
				"future_idx": future_index,
				"current_index": current_index,
				"future_index": future_index,
				"input_sequence_length": int(self.input_sequence_length),
				"prediction_horizon": int(self.prediction_horizon),
				"target_offset_from_start": int(target_offsets["target_offset_from_start"]),
				"target_offset_from_last_input": int(target_offsets["target_offset_from_last_input"]),
				"target_definition_version": target_definition_version(self.config),
				"current_file_path": str(current_file_path),
				"target_file_path": str(target_file_path),
				"input_channel_count_base": int(self.base_input_channel_count),
				"fuel_flux_engineered_channel_count": int(self.fuel_flux_engineered_channel_count),
				"atmospheric_engineered_channel_count": int(self.atmospheric_engineered_channel_count),
				"engineered_channel_count": int(self.engineered_channel_count),
				"total_input_channels": int(self.total_input_channels),
			}
			if self.use_patches:
				metadata["patch_top"] = int(patch_top)
				metadata["patch_left"] = int(patch_left)
				metadata["patch_size"] = int(self.patch_size)
			if self.energy_geometry is not None and self.task_type == "multitask" and target_array.shape[0] > 3:
				energy_target_raw = compute_energy_release_maps(
					future_tensor,
					config=self.config,
					area_2d_m2=np.asarray(self.energy_geometry["area_2d_m2"], dtype=np.float32),
				)["energy_release_total_MW"]
				if self.use_patches:
					energy_target_raw = energy_target_raw[patch_top:patch_top + self.patch_size, patch_left:patch_left + self.patch_size]
				metadata["geom_path"] = str(self.energy_geometry["geom_path"])
				metadata["terrain_path"] = str(self.energy_geometry["terrain_path"]) if self.energy_geometry["terrain_path"] is not None else None
				metadata["dy_m"] = float(self.energy_geometry["dy_m"])
				metadata["dx_min_m"] = float(self.energy_geometry["dx_min_m"])
				metadata["dx_max_m"] = float(self.energy_geometry["dx_max_m"])
				metadata["dx_mean_m"] = float(self.energy_geometry["dx_mean_m"])
				metadata["area_min_m2"] = float(self.energy_geometry["area_min_m2"])
				metadata["area_max_m2"] = float(self.energy_geometry["area_max_m2"])
				metadata["area_mean_m2"] = float(self.energy_geometry["area_mean_m2"])
				metadata["energy_target_transform"] = str(self.energy_release_config["target_transform"])
				metadata["energy_total_MW_min"] = float(np.min(energy_target_raw))
				metadata["energy_total_MW_max"] = float(np.max(energy_target_raw))
				metadata["energy_total_MW_mean"] = float(np.mean(energy_target_raw))
			return x_tensor, y_tensor, metadata

		return x_tensor, y_tensor


class MultiFireSequenceDataset(FireSequenceDataset):
	"""Sequence dataset that combines multiple independent wildfire time series."""

	def __init__(
		self,
		dataset_records: Sequence[Mapping[str, Any]],
		sample_refs: Sequence[Mapping[str, int]],
		input_sequence_length: int,
		prediction_horizon: int,
		target_channel: int,
		input_channel_count: int | None = None,
		input_channel_indices: Sequence[int] | None = None,
		task_type: str = "regression",
		fire_threshold: float = 0.5,
		use_patches: bool = False,
		patch_size: int = 64,
		active_patch_probability: float = 0.7,
		active_threshold: float = 0.0,
		normalization_stats: Mapping[str, np.ndarray] | str | Path | None = None,
		normalize_target: bool = False,
		transform=None,
		target_transform=None,
		return_metadata: bool = False,
		config: Mapping[str, Any] | None = None,
		split: str = "train",
	) -> None:
		self.dataset_records = [
			{
				**dict(record),
				"dataset_id": int(record["dataset_id"]),
				"dataset_name": str(record["dataset_name"]),
				"data_dir": Path(record["data_dir"]),
				"file_paths": _sort_chronologically(_as_path_list(record["file_paths"])),
				"num_files": int(record["num_files"]),
				"raw_shape": tuple(int(dimension) for dimension in record["raw_shape"]),
			}
			for record in dataset_records
		]
		if not self.dataset_records:
			raise ValueError("MultiFireSequenceDataset requires at least one dataset record.")

		self.input_sequence_length = int(input_sequence_length)
		self.prediction_horizon = int(prediction_horizon)
		self.target_channel = int(target_channel)
		self.input_channel_count = None if input_channel_count is None else int(input_channel_count)
		self.input_channel_indices = _coerce_index_list(input_channel_indices)
		self.task_type = str(task_type).lower()
		self.fire_threshold = float(fire_threshold)
		self.use_patches = bool(use_patches)
		self.patch_size = int(patch_size)
		self.active_patch_probability = float(active_patch_probability)
		self.active_threshold = float(active_threshold)
		self.transform = transform
		self.target_transform = target_transform
		self.return_metadata = bool(return_metadata)
		self.split = str(split).lower()
		self.config = dict(config) if isinstance(config, Mapping) else {
			"task_type": self.task_type,
			"target_channel": self.target_channel,
			"input_channel_count": self.input_channel_count,
			"input_channel_indices": self.input_channel_indices,
			"fire_threshold": self.fire_threshold,
			"active_threshold": self.active_threshold,
		}

		target_normalization_config = _resolve_target_normalization_config(self.config)
		self.normalize_target = bool(normalize_target or target_normalization_config["enabled"])

		if self.input_sequence_length <= 0:
			raise ValueError(f"input_sequence_length must be positive, got {self.input_sequence_length}.")
		if self.prediction_horizon < 0:
			raise ValueError(f"prediction_horizon must be non-negative, got {self.prediction_horizon}.")
		if not 0.0 <= self.active_patch_probability <= 1.0:
			raise ValueError("active_patch_probability must be in [0, 1], got {self.active_patch_probability}.")
		if self.patch_size <= 0:
			raise ValueError(f"patch_size must be positive, got {self.patch_size}.")
		if self.task_type not in {"regression", "segmentation", "multitask"}:
			raise ValueError(
				f"task_type must be 'regression', 'segmentation', or 'multitask', got {self.task_type!r}."
			)

		first_record = self.dataset_records[0]
		self.expected_height, self.expected_width, self.num_channels = first_record["raw_shape"]
		for record in self.dataset_records:
			record["temporal_trim"] = resolve_temporal_trim(record)
			record["effective_num_files"] = effective_num_frames(record)
			record_height, record_width, record_channels = record["raw_shape"]
			if record_channels != self.num_channels:
				raise ValueError(
					"All dataset records must share the same raw channel count. "
					f"Expected {self.num_channels}, got {record_channels} in {record['dataset_name']}."
				)
			if self.use_patches and (self.patch_size > record_height or self.patch_size > record_width):
				raise ValueError(
					"patch_size must be <= both spatial dimensions for every dataset. "
					f"Got patch_size={self.patch_size}, dataset={record['dataset_name']}, "
					f"raw_shape={record['raw_shape']}."
				)
			if not record["file_paths"]:
				raise ValueError(f"Dataset record {record['dataset_name']} has no file paths.")

		if self.input_channel_count is None:
			self.input_channel_count = self.num_channels
		if self.input_channel_count <= 0 or self.input_channel_count > self.num_channels:
			raise ValueError(f"input_channel_count must be in [1, {self.num_channels}], got {self.input_channel_count}.")
		if self.target_channel < 0 or self.target_channel >= self.num_channels:
			raise ValueError(f"target_channel must be in [0, {self.num_channels - 1}], got {self.target_channel}.")

		self.base_input_channel_indices = _resolve_input_channel_indices(self.config, self.input_channel_count)
		if any(index < 0 or index >= self.num_channels for index in self.base_input_channel_indices):
			raise ValueError(
				f"input_channel_indices must stay within [0, {self.num_channels - 1}], got {self.base_input_channel_indices}."
			)
		self.base_input_channel_count = len(self.base_input_channel_indices)
		self.fuel_flux_engineered_channel_count = _count_fuel_flux_engineered_channels(self.config)
		self.atmospheric_engineered_channel_count = count_atmospheric_engineered_channels(self.config)
		self.energy_release_config = resolve_energy_release_config(self.config)
		self.energy_target_channel_count = resolve_energy_target_count(self.config)
		self.energy_history_channel_count = 1 if self.energy_release_config["enabled"] and self.energy_release_config["add_as_input_history"] else 0
		self.engineered_channel_count = (
			self.fuel_flux_engineered_channel_count
			+ self.atmospheric_engineered_channel_count
			+ self.energy_history_channel_count
		)
		self.total_input_channels = self.base_input_channel_count + self.engineered_channel_count
		self.input_channels_after_engineering = self.total_input_channels
		self.engineered_feature_slices = resolve_engineered_feature_slices(self.config, self.base_input_channel_count)

		self.patching_config = resolve_patching_config(self.config)
		self.random_generator = np.random.default_rng()
		self.sample_refs = []
		for ref in sample_refs:
			normalized_ref: dict[str, Any] = {
				"dataset_id": int(ref["dataset_id"]),
				"dataset_name": str(ref.get("dataset_name", "")),
				"sample_index": int(ref["sample_index"]),
				"fire_split_group": str(ref.get("fire_split_group", self.split)),
			}
			if ref.get("patch") is not None:
				normalized_ref["patch"] = validate_patch_dict(ref["patch"])
			self.sample_refs.append(normalized_ref)
		for ref in self.sample_refs:
			dataset_id = int(ref["dataset_id"])
			if dataset_id < 0 or dataset_id >= len(self.dataset_records):
				raise ValueError(f"sample_refs contain invalid dataset_id={dataset_id}.")
			record = self.dataset_records[dataset_id]
			max_valid_start = int(record["effective_num_files"]) - self.input_sequence_length - self.prediction_horizon
			if ref["sample_index"] < 0 or ref["sample_index"] > max_valid_start:
				raise ValueError(
					"sample_refs contain invalid sample start positions. "
					f"Dataset={record['dataset_name']} valid range=[0, {max_valid_start}], "
					f"got {ref['sample_index']}."
				)
			patch = ref.get("patch")
			if patch is not None:
				patch = validate_patch_dict(patch)
				patch_height = int(patch["y1"] - patch["y0"])
				patch_width = int(patch["x1"] - patch["x0"])
				record_height, record_width = tuple(int(value) for value in record["raw_shape"][:2])
				if patch_height != self.patch_size or patch_width != self.patch_size:
					raise ValueError(
						f"Explicit patch size must match dataset patch_size={self.patch_size}. Got patch={patch}."
					)
				if patch["y0"] < 0 or patch["x0"] < 0 or patch["y1"] > record_height or patch["x1"] > record_width:
					raise ValueError(
						f"Explicit patch {patch} is out of bounds for dataset {record['dataset_name']} "
						f"with raw shape {record['raw_shape']}."
					)

		self.normalization_stats = FireSequenceDataset._coerce_normalization_stats(self, normalization_stats)
		self.input_normalization_on_device = bool(
			self.normalization_stats is not None and input_normalization_runs_on_device(self.config)
		)
		self.inputs_are_normalized = bool(
			self.normalization_stats is not None and not self.input_normalization_on_device
		)
		self.target_mean, self.target_std = FireSequenceDataset._resolve_target_normalization_stats(self)
		self.initial_fuel_maps = {
			int(record["dataset_id"]): _load_initial_fuel_map(
				record["file_paths"],
				self.config,
				initial_index=int(record["temporal_trim"]["trim_start_index"]),
			)
			for record in self.dataset_records
			if self.task_type == "multitask" or _resolve_engineered_features_config(self.config)["enabled"]
		}
		self.energy_geometries: dict[int, dict[str, Any]] = {}
		if self.energy_release_config["enabled"] and (self.task_type == "multitask" or self.energy_history_channel_count > 0):
			for record in self.dataset_records:
				dataset_id = int(record["dataset_id"])
				geometry = dict(record.get("geometry", {})) if isinstance(record.get("geometry"), Mapping) else {}
				if not geometry:
					height, width = tuple(int(value) for value in record["raw_shape"][:2])
					geometry_config = dict(self.config)
					if bool(record.get("geom_requires_transpose", False)):
						geometry_section = dict(geometry_config.get("geometry", {})) if isinstance(geometry_config.get("geometry"), Mapping) else {}
						geometry_section["allow_area_transpose_if_needed"] = True
						geometry_config["geometry"] = geometry_section
					geometry = load_fire_geometry(
						data_dir=Path(record["data_dir"]),
						config=geometry_config,
						expected_shape=(height, width),
					)
				self.energy_geometries[dataset_id] = geometry
				print(
					"Energy release geometry | "
					f"dataset={record['dataset_name']} geom={geometry['geom_path']} "
					f"area_min_m2={geometry['area_min_m2']:.6f} "
					f"area_mean_m2={geometry['area_mean_m2']:.6f} "
					f"area_max_m2={geometry['area_max_m2']:.6f}"
				)

	def __len__(self) -> int:
		return len(self.sample_refs)

	def _validate_tensor_shape_for_record(self, tensor: np.ndarray, file_path: Path, dataset_record: Mapping[str, Any]) -> None:
		"""Ensure one loaded tensor matches the expected shape for its own dataset."""

		expected_shape = tuple(int(dimension) for dimension in dataset_record["raw_shape"])
		if tensor.ndim != 3:
			raise ValueError(f"Expected a 3D tensor in {file_path}, got shape {tensor.shape}.")
		if tuple(int(dimension) for dimension in tensor.shape) != expected_shape:
			raise ValueError(
				f"Inconsistent tensor shape in {file_path}. "
				f"Expected {expected_shape}, got {tuple(int(dimension) for dimension in tensor.shape)}."
			)

	def _build_activity_map_for_patching(self, target_array: np.ndarray) -> np.ndarray:
		"""Build the 2D activity map used for active patch sampling."""

		if self.task_type != "multitask":
			return np.asarray(target_array, dtype=np.float32)

		active_source = str(self.patching_config["active_source"])
		if active_source == "mask":
			return np.asarray(target_array[2], dtype=np.float32)
		if active_source == "step_consumed_fuel":
			return np.maximum(np.asarray(target_array[0], dtype=np.float32), np.asarray(target_array[1], dtype=np.float32))
		if active_source == "energy_release" and target_array.shape[0] > 3:
			energy_map = np.asarray(target_array[3], dtype=np.float32)
			energy_threshold = float(
				transform_energy_target(
					np.asarray([[self.patching_config["energy_active_threshold_MW"]]], dtype=np.float32),
					self.config,
				)[0, 0]
			)
			return (energy_map > energy_threshold).astype(np.float32, copy=False)
		if active_source == "combined_target":
			combined = np.asarray(target_array[2], dtype=np.float32).copy()
			consumed_threshold = float(self.patching_config["consumed_active_threshold"])
			combined = np.maximum(
				combined,
				(np.asarray(target_array[0], dtype=np.float32) > consumed_threshold).astype(np.float32, copy=False),
			)
			combined = np.maximum(
				combined,
				(np.asarray(target_array[1], dtype=np.float32) > consumed_threshold).astype(np.float32, copy=False),
			)
			if target_array.shape[0] > 3:
				energy_threshold = float(
					transform_energy_target(
						np.asarray([[self.patching_config["energy_active_threshold_MW"]]], dtype=np.float32),
						self.config,
					)[0, 0]
				)
				combined = np.maximum(
					combined,
					(np.asarray(target_array[3], dtype=np.float32) > energy_threshold).astype(np.float32, copy=False),
				)
			return combined.astype(np.float32, copy=False)
		return np.asarray(target_array[2], dtype=np.float32)

	def _select_patch_for_ref(self, ref: Mapping[str, Any], target_array: np.ndarray, dataset_record: Mapping[str, Any]) -> dict[str, int] | None:
		"""Resolve either an explicit patch or a sampled train-time patch."""

		explicit_patch = ref.get("patch")
		if explicit_patch is not None:
			return validate_patch_dict(explicit_patch)
		if not self.use_patches:
			return None

		height, width = tuple(int(value) for value in dataset_record["raw_shape"][:2])
		activity_map = self._build_activity_map_for_patching(target_array)
		active_patch = sample_active_patch(
			activity_map=(activity_map > 0).astype(np.float32, copy=False),
			patch_h=self.patch_size,
			patch_w=self.patch_size,
			rng=self.random_generator,
			min_active_pixels=int(self.patching_config["min_active_pixels"]),
			center_on_active_pixel=bool(self.patching_config["center_on_active_pixel"]),
			jitter_active_center=bool(self.patching_config["jitter_active_center"]),
			max_center_jitter_pixels=int(self.patching_config["max_center_jitter_pixels"]),
		)
		active_probability = float(self.patching_config["active_patch_probability"])
		if active_patch is not None and self.random_generator.random() < active_probability:
			return active_patch
		return sample_random_patch(
			height=height,
			width=width,
			patch_h=self.patch_size,
			patch_w=self.patch_size,
			rng=self.random_generator,
		)

	def __getitem__(self, index: int):
		if torch is None:
			raise ImportError("PyTorch is required to index MultiFireSequenceDataset.")

		ref = self.sample_refs[index]
		dataset_record = self.dataset_records[int(ref["dataset_id"])]
		file_paths = dataset_record["file_paths"]
		local_sample_start = int(ref["sample_index"])
		sample_meta = temporal_sample_metadata(
			dataset_record,
			local_start_idx=local_sample_start,
			input_sequence_length=self.input_sequence_length,
			prediction_horizon=self.prediction_horizon,
		)
		sample_start = int(sample_meta["original_start_idx"])
		current_index = int(sample_meta["original_last_input_idx"])
		future_index = int(sample_meta["original_target_idx"])
		input_file_paths = [file_paths[original_index] for original_index in sample_meta["original_input_indices"]]
		current_file_path = file_paths[current_index]
		target_file_path = file_paths[future_index]

		raw_input_frames: list[np.ndarray] = []
		base_input_frames: list[np.ndarray] = []
		for file_path in input_file_paths:
			tensor = FireSequenceDataset._load_tensor(self, file_path)
			self._validate_tensor_shape_for_record(tensor, file_path, dataset_record)
			raw_frame = np.asarray(tensor, dtype=np.float32)
			raw_input_frames.append(raw_frame)
			base_input_frames.append(_slice_channels(raw_frame, self.base_input_channel_indices))

		current_tensor = raw_input_frames[-1]
		future_tensor = FireSequenceDataset._load_tensor(self, target_file_path)
		self._validate_tensor_shape_for_record(future_tensor, target_file_path, dataset_record)
		future_tensor = np.asarray(future_tensor, dtype=np.float32)

		if self.task_type == "multitask":
			initial_fuel_map = self.initial_fuel_maps.get(int(dataset_record["dataset_id"]))
			if initial_fuel_map is None:
				raise ValueError("initial_fuel_map must be available for multitask targets.")
			energy_geometry = self.energy_geometries.get(int(dataset_record["dataset_id"]))
			target_array = build_multitask_target(
				current_frame=current_tensor,
				future_frame=future_tensor,
				initial_fuel=initial_fuel_map,
				config=self.config,
				energy_geometry=energy_geometry,
			)
			target_map_for_sampling = np.asarray(target_array[2], dtype=np.float32)
		else:
			energy_geometry = self.energy_geometries.get(int(dataset_record["dataset_id"]))
			raw_target_array = np.asarray(future_tensor[:, :, self.target_channel], dtype=np.float32)
			target_array = raw_target_array.copy()
			if self.task_type == "segmentation":
				target_array = (target_array > self.fire_threshold).astype(np.float32, copy=False)

		stacked_inputs = np.stack(base_input_frames, axis=0).astype(np.float32, copy=False)
		engineered_inputs = build_engineered_features(
			input_frames=np.stack(raw_input_frames, axis=0).astype(np.float32, copy=False),
			file_paths=file_paths,
			start_index=sample_start,
			config=self.config,
			energy_geometry=energy_geometry,
			initial_fuel_index=int(sample_meta["trim_start_index"]),
		)
		if engineered_inputs.shape[:3] != stacked_inputs.shape[:3]:
			raise ValueError(
				"Engineered feature tensor must align with base inputs in (T, H, W). "
				f"Got base={stacked_inputs.shape} engineered={engineered_inputs.shape}."
			)
		if engineered_inputs.shape[-1] != self.engineered_channel_count:
			raise ValueError(
				f"Expected {self.engineered_channel_count} engineered channels, got {engineered_inputs.shape[-1]}."
			)
		if engineered_inputs.shape[-1] > 0:
			stacked_inputs = np.concatenate([stacked_inputs, engineered_inputs], axis=-1)

		resolved_patch = self._select_patch_for_ref(ref, target_array, dataset_record)
		if resolved_patch is not None:
			stacked_inputs = extract_patch_array(stacked_inputs, resolved_patch)
			if self.task_type == "multitask":
				target_array = np.transpose(
					extract_patch_array(np.transpose(target_array, (1, 2, 0)), resolved_patch),
					(2, 0, 1),
				)
			else:
				target_array = extract_patch_array(target_array, resolved_patch)

		stacked_inputs = FireSequenceDataset._normalize_inputs(self, stacked_inputs)
		target_array = FireSequenceDataset._normalize_target(self, target_array)

		stacked_inputs = np.transpose(stacked_inputs, (0, 3, 1, 2))
		stacked_inputs = np.ascontiguousarray(stacked_inputs, dtype=np.float32)
		if self.task_type == "multitask":
			target_array = np.ascontiguousarray(target_array, dtype=np.float32)
			if not np.all(np.isin(np.unique(target_array[2]), np.asarray([0.0, 1.0], dtype=np.float32))):
				raise ValueError("Multitask mask channel must contain only 0.0 and 1.0 after processing.")
			if target_array.shape[0] != _multitask_output_channel_count(self.config):
				raise ValueError(
					f"Expected multitask target shape ({_multitask_output_channel_count(self.config)}, H, W), got {target_array.shape}."
				)
		else:
			target_array = np.expand_dims(np.ascontiguousarray(target_array, dtype=np.float32), axis=0)

		x_tensor = torch.from_numpy(stacked_inputs).to(torch.float32)
		y_tensor = torch.from_numpy(target_array).to(torch.float32)
		if self.transform is not None:
			x_tensor = self.transform(x_tensor)
		if self.target_transform is not None:
			y_tensor = self.target_transform(y_tensor)

		if self.return_metadata:
			input_indices = list(range(sample_start, sample_start + self.input_sequence_length))
			target_offsets = temporal_target_offsets(
				{
					"input_sequence_length": self.input_sequence_length,
					"prediction_horizon": self.prediction_horizon,
				}
			)
			metadata = {
				"dataset_id": int(dataset_record["dataset_id"]),
				"dataset_name": str(dataset_record["dataset_name"]),
				"data_dir": str(dataset_record["data_dir"]),
				"sample_index": local_sample_start,
				"start_idx": local_sample_start,
				"input_indices": input_indices,
				"last_input_idx": current_index,
				"target_idx": future_index,
				"current_idx": current_index,
				"future_idx": future_index,
				"current_index": current_index,
				"future_index": future_index,
				"input_sequence_length": int(self.input_sequence_length),
				"prediction_horizon": int(self.prediction_horizon),
				"target_offset_from_start": int(target_offsets["target_offset_from_start"]),
				"target_offset_from_last_input": int(target_offsets["target_offset_from_last_input"]),
				"target_definition_version": target_definition_version(self.config),
				"local_start_idx": int(sample_meta["local_start_idx"]),
				"local_input_indices": list(sample_meta["local_input_indices"]),
				"local_last_input_idx": int(sample_meta["local_last_input_idx"]),
				"local_target_idx": int(sample_meta["local_target_idx"]),
				"original_start_idx": int(sample_meta["original_start_idx"]),
				"original_input_indices": list(sample_meta["original_input_indices"]),
				"original_last_input_idx": int(sample_meta["original_last_input_idx"]),
				"original_target_idx": int(sample_meta["original_target_idx"]),
				"trim_start_index": int(sample_meta["trim_start_index"]),
				"trim_end_index": int(sample_meta["trim_end_index"]),
				"trimmed_num_frames": int(sample_meta["trimmed_num_frames"]),
				"original_num_frames": int(sample_meta["original_num_frames"]),
				"temporal_trim_enabled": bool(sample_meta["temporal_trim_enabled"]),
				"current_file": str(current_file_path),
				"future_file": str(target_file_path),
				"current_file_path": str(current_file_path),
				"target_file_path": str(target_file_path),
				"input_channel_count_base": int(self.base_input_channel_count),
				"fuel_flux_engineered_channel_count": int(self.fuel_flux_engineered_channel_count),
				"atmospheric_engineered_channel_count": int(self.atmospheric_engineered_channel_count),
				"engineered_channel_count": int(self.engineered_channel_count),
				"total_input_channels": int(self.total_input_channels),
			}
			metadata["split"] = self.split
			metadata["fire_split_group"] = str(ref.get("fire_split_group", self.split))
			if resolved_patch is not None:
				metadata["patch"] = dict(resolved_patch)
				metadata["patch_top"] = int(resolved_patch["y0"])
				metadata["patch_left"] = int(resolved_patch["x0"])
				metadata["patch_bottom"] = int(resolved_patch["y1"])
				metadata["patch_right"] = int(resolved_patch["x1"])
				metadata["patch_size"] = int(self.patch_size)
			if energy_geometry is not None and self.task_type == "multitask" and target_array.shape[0] > 3:
				energy_target_raw = compute_energy_release_maps(
					future_tensor,
					config=self.config,
					area_2d_m2=np.asarray(energy_geometry["area_2d_m2"], dtype=np.float32),
				)["energy_release_total_MW"]
				if resolved_patch is not None:
					energy_target_raw = extract_patch_array(energy_target_raw, resolved_patch)
				metadata["geom_path"] = str(energy_geometry["geom_path"])
				metadata["terrain_path"] = str(energy_geometry["terrain_path"]) if energy_geometry["terrain_path"] is not None else None
				metadata["dy_m"] = float(energy_geometry["dy_m"])
				metadata["dx_min_m"] = float(energy_geometry["dx_min_m"])
				metadata["dx_max_m"] = float(energy_geometry["dx_max_m"])
				metadata["dx_mean_m"] = float(energy_geometry["dx_mean_m"])
				metadata["area_min_m2"] = float(energy_geometry["area_min_m2"])
				metadata["area_max_m2"] = float(energy_geometry["area_max_m2"])
				metadata["area_mean_m2"] = float(energy_geometry["area_mean_m2"])
				metadata["energy_target_transform"] = str(self.energy_release_config["target_transform"])
				metadata["energy_total_MW_min"] = float(np.min(energy_target_raw))
				metadata["energy_total_MW_max"] = float(np.max(energy_target_raw))
				metadata["energy_total_MW_mean"] = float(np.mean(energy_target_raw))
			return x_tensor, y_tensor, metadata

		return x_tensor, y_tensor


class MultiFirePatchSequenceDataset(MultiFireSequenceDataset):
	"""Backward-compatible explicit name for the multi-fire patch-aware dataset."""


def _resolve_dataloader_options(config: Mapping[str, Any], split: str) -> dict[str, Any]:
	"""Resolve split-specific DataLoader options with legacy fallbacks."""

	training_config = _get_section(config, "training")
	performance_config = get_performance_config(config)
	data_loader_config = _get_section(config, "data_loader")
	split_key = str(split).lower()
	split_config = data_loader_config.get(split_key, {})
	if not isinstance(split_config, Mapping):
		split_config = {}
	if split_key == "test":
		val_config = data_loader_config.get("val", {})
		if isinstance(val_config, Mapping):
			merged_split_config = dict(val_config)
			merged_split_config.update(dict(split_config))
			split_config = merged_split_config

	batch_size = int(
		split_config.get(
			"batch_size",
			data_loader_config.get("batch_size", training_config.get("batch_size", config.get("batch_size", 4))),
		)
	)
	raw_num_workers = split_config.get(
		"num_workers",
		data_loader_config.get("num_workers", training_config.get("num_workers", config.get("num_workers", 0))),
	)
	num_workers = cap_num_workers_by_slurm(config, raw_num_workers)
	pin_memory_default = bool(torch.cuda.is_available()) if torch is not None else False
	pin_memory = bool(
		split_config.get(
			"pin_memory",
			data_loader_config.get("pin_memory", training_config.get("pin_memory", pin_memory_default)),
		)
	)
	drop_last_default = False
	drop_last = bool(split_config.get("drop_last", data_loader_config.get("drop_last", drop_last_default)))
	options: dict[str, Any] = {
		"batch_size": max(1, batch_size),
		"num_workers": num_workers,
		"pin_memory": pin_memory,
		"drop_last": drop_last,
	}
	if num_workers > 0:
		options["persistent_workers"] = bool(
			split_config.get(
				"persistent_workers",
				data_loader_config.get("persistent_workers", training_config.get("persistent_workers", False)),
			)
		)
		prefetch_factor = split_config.get(
			"prefetch_factor",
			data_loader_config.get("prefetch_factor", training_config.get("prefetch_factor", None)),
		)
		if prefetch_factor not in (None, "", "null"):
			options["prefetch_factor"] = max(1, int(prefetch_factor))
		multiprocessing_context = split_config.get(
			"multiprocessing_context",
			data_loader_config.get("multiprocessing_context", training_config.get("multiprocessing_context", None)),
		)
		if multiprocessing_context not in (None, "", "null"):
			options["multiprocessing_context"] = str(multiprocessing_context)
	else:
		options["persistent_workers"] = False
	if not bool(performance_config.get("non_blocking_transfer", True)):
		options["pin_memory"] = False
	return options


def _resolve_normalization_stats_path(
	config: Mapping[str, Any],
	config_path: Path | None,
) -> Path | None:
	"""Resolve the configured normalization stats path when it exists."""

	normalization_config = _get_section(config, "normalization")
	if bool(normalization_config.get("enabled", True)) is False:
		return None
	normalization_path = normalization_config.get("stats_path", normalization_config.get("path"))
	if not normalization_path:
		if bool(normalization_config.get("require_stats", False)):
			raise FileNotFoundError("normalization.require_stats=true, but normalization.path/stats_path is not configured.")
		return None
	resolved_normalization_path = _resolve_path(config_path, normalization_path)
	if resolved_normalization_path.exists():
		return resolved_normalization_path
	if bool(normalization_config.get("require_stats", False)):
		raise FileNotFoundError(
			f"Normalization stats file not found: {resolved_normalization_path}\n"
			"Compute train-only input stats first, for example:\n"
			"python scripts/compute_normalization.py --config configs/default.yaml --from_cache"
		)
	return None


def _build_cached_dataloaders(
	config: Mapping[str, Any],
	normalization_stats: Path | None,
):
	"""Build DataLoaders backed by precomputed patch-cache shards."""

	from src.data.cache import get_patch_cache_dir, validate_patch_cache
	from src.data.cached_patch_dataset import CachedPatchDataset, CachedShardBatchSampler

	cache_config = _get_section(config, "cache")
	cache_dir = get_patch_cache_dir(config)
	try:
		cache_summary = validate_patch_cache(config, split=["train", "val", "test"])
	except Exception as exc:
		if bool(cache_config.get("allow_dynamic_fallback", False)):
			print(f"WARNING: patch-cache validation failed; falling back to dynamic Dataset: {exc}")
			return None
		raise

	if not bool(cache_config.get("save_normalized_inputs", False)) and normalization_stats is None:
		raise RuntimeError(
			"cache.use_precomputed_patches=true and cache.save_normalized_inputs=false, "
			"but normalization stats were not found. Run:\n"
			"python scripts/compute_normalization.py --config configs/default.yaml --from_cache"
		)

	return_metadata = bool(config.get("return_metadata", False))
	train_dataset = CachedPatchDataset(
		cache_dir=cache_dir,
		split="train",
		config=config,
		normalization_stats=normalization_stats,
		return_metadata=return_metadata,
	)
	val_dataset = CachedPatchDataset(
		cache_dir=cache_dir,
		split="val",
		config=config,
		normalization_stats=normalization_stats,
		return_metadata=return_metadata,
	)
	test_dataset = CachedPatchDataset(
		cache_dir=cache_dir,
		split="test",
		config=config,
		normalization_stats=normalization_stats,
		return_metadata=return_metadata,
	)

	train_dataloader_options = _resolve_dataloader_options(config, "train")
	val_dataloader_options = _resolve_dataloader_options(config, "val")
	test_dataloader_options = _resolve_dataloader_options(config, "test")
	sampler = None
	dataset_sampling = _get_section(config, "dataset_sampling")
	if bool(dataset_sampling.get("enabled", False)):
		sampling_mode = str(dataset_sampling.get("mode", "uniform_samples")).lower()
		if sampling_mode in {"balanced_by_dataset", "balanced_by_fire"}:
			if WeightedRandomSampler is None:
				print("WARNING: WeightedRandomSampler is unavailable; falling back to shuffled cached training.")
			elif not train_dataset.metadata:
				print("WARNING: cached train metadata is unavailable; falling back to shuffled cached training.")
			else:
				counts_by_dataset: dict[int, int] = {}
				for item in train_dataset.metadata:
					dataset_id = int(item.get("dataset_id", -1))
					counts_by_dataset[dataset_id] = counts_by_dataset.get(dataset_id, 0) + 1
				weights = [1.0 / counts_by_dataset[int(item.get("dataset_id", -1))] for item in train_dataset.metadata]
				sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
		elif sampling_mode != "uniform_samples":
			print(
				"WARNING: Unsupported dataset_sampling.mode for cached loading; "
				f"falling back to normal shuffle: {sampling_mode!r}"
			)

	train_batch_sampler_mode = str(cache_config.get("train_batch_sampler", "weighted_random")).lower()
	if train_batch_sampler_mode in {"shard_local", "shard_local_shuffle", "shard_local_random"}:
		if sampler is not None:
			print(
				"Using cached shard-local train batching; dataset_sampling weighted sampler is disabled "
				"for this cached loader so batches stay local to cache shards."
			)
		batch_size = int(train_dataloader_options["batch_size"])
		drop_last = bool(train_dataloader_options.get("drop_last", False))
		loader_options = dict(train_dataloader_options)
		loader_options.pop("batch_size", None)
		loader_options.pop("drop_last", None)
		train_batch_sampler = CachedShardBatchSampler(
			train_dataset,
			batch_size=batch_size,
			drop_last=drop_last,
			shuffle_shards=train_batch_sampler_mode != "shard_local",
			shuffle_within_shard=train_batch_sampler_mode != "shard_local",
			seed=int(_get_section(config, "training").get("seed", config.get("seed", 42))),
		)
		train_loader = DataLoader(
			train_dataset,
			batch_sampler=train_batch_sampler,
			**loader_options,
		)
	else:
		train_loader = DataLoader(
			train_dataset,
			shuffle=sampler is None,
			sampler=sampler,
			**train_dataloader_options,
		)
	val_loader = DataLoader(
		val_dataset,
		shuffle=False,
		**val_dataloader_options,
	)
	test_loader = DataLoader(
		test_dataset,
		shuffle=False,
		**test_dataloader_options,
	)
	print(
		"Using precomputed patch cache | "
		f"cache_dir={cache_dir} "
		f"train={cache_summary['splits']['train']['num_samples']} "
		f"val={cache_summary['splits']['val']['num_samples']} "
		f"test={cache_summary['splits']['test']['num_samples']}"
	)
	return train_loader, val_loader, test_loader


def create_dataloaders(config):
	"""Build train/validation/test DataLoaders from a configuration dictionary."""

	if torch is None or DataLoader is None:
		raise ImportError("PyTorch is required to build DataLoaders for wildfire forecasting.")

	dataloader_config = _get_section(config, "dataloader")
	if str(dataloader_config.get("source", "")).lower() == "processed_full_frames":
		config_path_value = config.get("config_path", config.get("_config_path"))
		config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
		processed = _get_section(config, "processed_dataset")
		root_value = dataloader_config.get("dataset_root", processed.get("root", "/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1"))
		root = _resolve_path(config_path, root_value)
		pattern = str(dataloader_config.get("sample_pattern", "consecutive5_h10"))
		sample_dir_value = dataloader_config.get("split_sample_index_dir")
		sample_dir = _resolve_path(config_path, sample_dir_value) if sample_dir_value else root / "indices" / "temporal"
		sample_path_value = dataloader_config.get("sample_index_path")
		sample_path = _resolve_path(config_path, sample_path_value) if sample_path_value else sample_dir / f"samples_{pattern}.jsonl"
		normalization = _get_section(config, "normalization")
		normalize_inputs = bool(dataloader_config.get("normalize_inputs", True))
		stats_value = normalization.get("stats_path") if normalize_inputs else None
		stats_path = _resolve_path(config_path, stats_value) if stats_value else None
		if normalize_inputs and bool(normalization.get("require_stats", False)) and (stats_path is None or not stats_path.exists()):
			raise FileNotFoundError(f"Processed-full-frame normalization stats are required but missing: {stats_path}")
		return_terrain_value = dataloader_config.get("return_terrain", "auto")
		architecture = str(_get_section(config, "model").get("architecture", "")).lower()
		cawfe_config = _get_section(config, "cawfe_latte")
		cawfe_v11_config = _get_section(config, "cawfe_latte_v1_1")
		terrain_conditioning_enabled = bool(cawfe_v11_config.get("use_terrain_conditioning", cawfe_config.get("use_terrain_conditioning", False))) if architecture in {"cawfe_latte_v1_1", "cawfe_latte_v1_2"} else bool(cawfe_config.get("use_terrain_conditioning", False))
		return_terrain = bool(return_terrain_value) if not isinstance(return_terrain_value, str) or return_terrain_value.lower() != "auto" else (architecture in {"cawfe_latte", "cawfe_latte_v1_1", "cawfe_latte_v1_2"} and terrain_conditioning_enabled)
		common = {"dataset_root": root, "sample_index_path": sample_path, "normalization_stats_path": stats_path, "normalize_inputs": normalize_inputs, "input_key": str(dataloader_config.get("input_key", "x_engineered")), "return_metadata": bool(config.get("return_metadata", dataloader_config.get("return_metadata", False))), "single_frame_mode": str(dataloader_config.get("single_frame_mode", "as_is")), "repeat_to_length": dataloader_config.get("repeat_to_length"), "return_terrain": return_terrain, "terrain_key": str(dataloader_config.get("terrain_key", "terrain_features"))}
		train_dataset = ProcessedTemporalPatchDataset(split="train", **common)
		val_dataset = ProcessedTemporalPatchDataset(split="val", **common)
		test_dataset = ProcessedTemporalPatchDataset(split="test", **common)
		fire_sets = {split: {str(record["fire_name"]) for record in dataset.records} for split, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset))}
		for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
			overlap = fire_sets[left] & fire_sets[right]
			if overlap:
				raise ValueError(f"Processed dataset fire split leakage between {left} and {right}: {sorted(overlap)}")
		for split, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
			bad = [record.get("sample_id", "<unknown>") for record in dataset.records if record.get("split") != split]
			if bad:
				raise ValueError(f"Processed {split} dataset contains records from another split: {bad[:3]}")
		options = [_resolve_dataloader_options(config, split) for split in ("train", "val", "test")]
		loaders = (DataLoader(train_dataset, shuffle=True, **options[0]), DataLoader(val_dataset, shuffle=False, **options[1]), DataLoader(test_dataset, shuffle=False, **options[2]))
		print(f"Data source: processed_full_frames | root={root} | pattern={pattern} | sample_index={sample_path} | normalization={stats_path}")
		print(f"Processed samples | train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")
		return loaders

	for required_key in ("file_pattern", "input_sequence_length", "prediction_horizon"):
		if required_key not in config:
			raise KeyError(f"Config is missing required key '{required_key}'.")
	if "data_dir" not in config and "data_dirs" not in config:
		raise KeyError("Config must define either data_dir or data_dirs.")

	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	file_pattern = str(config["file_pattern"])

	input_sequence_length = int(config["input_sequence_length"])
	prediction_horizon = int(config["prediction_horizon"])
	target_channel = int(config.get("target_channel", 0))
	input_channel_count = int(config.get("input_channel_count", config.get("model", {}).get("input_channels", 0)))
	if input_channel_count <= 0:
		raise KeyError("Config must define a positive input_channel_count or model.input_channels.")
	split_mode = str(config.get("split_mode", "train_val_test")).lower()
	if split_mode == "multi_dataset_chronological":
		split_mode = "multi_fire_chronological"
	train_fraction = float(config.get("train_fraction", 0.7))
	val_fraction = float(config.get("val_fraction", 0.15))
	test_fraction = float(config.get("test_fraction", 0.15))

	task_type = str(config.get("task_type", _get_section(config, "training").get("task_type", "regression"))).lower()
	target_normalization = _resolve_target_normalization_config(config)
	patching_config = resolve_patching_config(config)
	patch_size = _resolve_square_patch_size_from_config(config)
	use_train_patches = bool(patching_config["enabled"])
	use_eval_patches = bool(patching_config["enabled"] and patching_config["eval_mode"] == "sliding_window")
	return_metadata_for_multi = bool(config.get("return_metadata", False))

	normalization_stats = None
	normalization_stats = _resolve_normalization_stats_path(config, config_path)

	cache_config = _get_section(config, "cache")
	if bool(cache_config.get("enabled", False)) and bool(cache_config.get("use_precomputed_patches", False)):
		cached_loaders = _build_cached_dataloaders(config, normalization_stats)
		if cached_loaders is not None:
			return cached_loaders

	if split_mode in {"multi_fire_chronological", "manual_fire_holdout"}:
		data_dirs = resolve_data_dirs(config)
		if "data_dirs" in config and config.get("data_dirs"):
			print(f"Using data_dirs mode with {len(data_dirs)} datasets.")
		else:
			print("Using legacy data_dir fallback as a single-item data_dirs list.")
		dataset_records = discover_multiple_datasets(config)
		if split_mode == "manual_fire_holdout":
			manual_section = _get_section(config, "manual_fire_split")
			sample_refs = manual_fire_holdout_splits(
				dataset_records=dataset_records,
				train_fire_names=manual_section.get("train_fires", []),
				val_fire_names=manual_section.get("val_fires", []),
				test_fire_names=manual_section.get("test_fires", []),
				input_sequence_length=input_sequence_length,
				prediction_horizon=prediction_horizon,
				config=config,
			)
		else:
			sample_refs = multi_fire_chronological_splits(
				dataset_records=dataset_records,
				input_sequence_length=input_sequence_length,
				prediction_horizon=prediction_horizon,
				train_fraction=train_fraction,
				val_fraction=val_fraction,
				test_fraction=test_fraction,
			)
		normalization_stats = _resolve_normalization_stats_path(config, config_path)

		common_multi_kwargs = {
			"dataset_records": dataset_records,
			"input_sequence_length": input_sequence_length,
			"prediction_horizon": prediction_horizon,
			"target_channel": target_channel,
			"input_channel_count": input_channel_count,
			"input_channel_indices": config.get("input_channel_indices"),
			"task_type": task_type,
			"fire_threshold": float(config.get("fire_threshold", _get_section(config, "training").get("fire_threshold", 0.5))),
			"patch_size": int(patch_size),
			"active_patch_probability": float(patching_config["active_patch_probability"]),
			"active_threshold": float(config.get("active_threshold", config.get("fire_threshold", 0.5))),
			"normalization_stats": normalization_stats,
			"normalize_target": bool(target_normalization["enabled"]),
			"return_metadata": return_metadata_for_multi,
			"config": config,
		}
		train_refs = sample_refs["train"]
		val_refs = sample_refs["val"]
		test_refs = sample_refs["test"]
		train_patch_mode = resolve_split_patch_mode(config, "train", prefer_cache=False)
		val_patch_mode = resolve_split_patch_mode(config, "val", prefer_cache=False)
		test_patch_mode = resolve_split_patch_mode(config, "test", prefer_cache=False)
		if train_patch_mode == "sliding_window":
			train_refs = build_sliding_patch_refs_for_split(dataset_records=dataset_records, sample_refs=train_refs, split="train", config=config)
		if val_patch_mode == "sliding_window":
			val_refs = build_sliding_patch_refs_for_split(dataset_records=dataset_records, sample_refs=val_refs, split="val", config=config)
		if test_patch_mode == "sliding_window":
			test_refs = build_sliding_patch_refs_for_split(dataset_records=dataset_records, sample_refs=test_refs, split="test", config=config)
		train_dataset = MultiFirePatchSequenceDataset(
			sample_refs=train_refs,
			use_patches=use_train_patches,
			split="train",
			**common_multi_kwargs,
		)
		val_dataset = MultiFirePatchSequenceDataset(
			sample_refs=val_refs,
			use_patches=bool(val_patch_mode == "sliding_window"),
			split="val",
			**common_multi_kwargs,
		)
		test_dataset = MultiFirePatchSequenceDataset(
			sample_refs=test_refs,
			use_patches=bool(test_patch_mode == "sliding_window"),
			split="test",
			**common_multi_kwargs,
		)

		train_dataloader_options = _resolve_dataloader_options(config, "train")
		val_dataloader_options = _resolve_dataloader_options(config, "val")
		test_dataloader_options = _resolve_dataloader_options(config, "test")
		sampler = None
		dataset_sampling = _get_section(config, "dataset_sampling")
		if bool(dataset_sampling.get("enabled", False)):
			sampling_mode = str(dataset_sampling.get("mode", "uniform_samples")).lower()
			if sampling_mode in {"balanced_by_dataset", "balanced_by_fire"}:
				if WeightedRandomSampler is None:
					print("WARNING: WeightedRandomSampler is unavailable; falling back to normal shuffled training.")
				else:
					counts_by_dataset: dict[int, int] = {}
					for ref in train_dataset.sample_refs:
						dataset_id = int(ref["dataset_id"])
						counts_by_dataset[dataset_id] = counts_by_dataset.get(dataset_id, 0) + 1
					weights = [1.0 / counts_by_dataset[int(ref["dataset_id"])] for ref in train_dataset.sample_refs]
					for dataset_id, count in sorted(counts_by_dataset.items()):
						print(
							f"Train fire sampling weight | dataset={train_dataset.dataset_records[dataset_id]['dataset_name']} "
							f"samples={count} weight={1.0 / count:.6f}"
						)
					sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
			elif sampling_mode != "uniform_samples":
				print(
					"WARNING: Unsupported dataset_sampling.mode for multi-fire loading; "
					f"falling back to normal shuffle: {sampling_mode!r}"
				)

		train_loader = DataLoader(
			train_dataset,
			shuffle=sampler is None,
			sampler=sampler,
			**train_dataloader_options,
		)
		val_loader = DataLoader(
			val_dataset,
			shuffle=False,
			**val_dataloader_options,
		)
		test_loader = DataLoader(
			test_dataset,
			shuffle=False,
			**test_dataloader_options,
		)
		return train_loader, val_loader, test_loader

	data_dir = _resolve_path(config_path, config["data_dir"])
	if not data_dir.exists():
		raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
	files = discover_dataset_files(data_dir, file_pattern)
	if split_mode == "train_val_external_test":
		split_indices = {
			**chronological_train_val_split_indices(
				num_timesteps=len(files),
				input_sequence_length=input_sequence_length,
				prediction_horizon=prediction_horizon,
				train_fraction=train_fraction,
				val_fraction=val_fraction,
			),
			"test": [],
		}
	else:
		split_indices = chronological_split_indices(
			num_timesteps=len(files),
			input_sequence_length=input_sequence_length,
			prediction_horizon=prediction_horizon,
			train_fraction=train_fraction,
			val_fraction=val_fraction,
			test_fraction=test_fraction,
			split_mode=split_mode,
		)
	dataset_kwargs = {
		"file_paths": files,
		"input_sequence_length": input_sequence_length,
		"prediction_horizon": prediction_horizon,
		"target_channel": target_channel,
		"input_channel_count": input_channel_count,
		"input_channel_indices": config.get("input_channel_indices"),
		"task_type": task_type,
		"fire_threshold": float(config.get("fire_threshold", _get_section(config, "training").get("fire_threshold", 0.5))),
		"patch_size": int(patch_size),
		"active_patch_probability": float(patching_config["active_patch_probability"]),
		"active_threshold": float(config.get("active_threshold", config.get("fire_threshold", 0.5))),
		"normalization_stats": normalization_stats,
		"normalize_target": bool(target_normalization["enabled"]),
		"config": config,
	}

	train_dataset = FireSequenceDataset(sample_indices=split_indices["train"], use_patches=use_train_patches, **dataset_kwargs)
	val_dataset = FireSequenceDataset(sample_indices=split_indices["val"], use_patches=use_eval_patches, **dataset_kwargs)
	test_dataset = None
	test_data_dir_value = config.get("test_data_dir")
	if split_mode == "train_val_external_test":
		if test_data_dir_value not in (None, "", "null"):
			test_data_dir = _resolve_path(config_path, test_data_dir_value)
			external_test_file_pattern = str(config.get("external_test_file_pattern", file_pattern))
			if not test_data_dir.exists():
				raise FileNotFoundError(f"External test data directory does not exist: {test_data_dir}")
			external_test_files = _sort_chronologically(list(test_data_dir.glob(external_test_file_pattern)))
			if not external_test_files:
				raise FileNotFoundError(
					f"No external test files found in '{test_data_dir}' using pattern '{external_test_file_pattern}'."
				)
			external_test_dataset_kwargs = {**dataset_kwargs, "file_paths": external_test_files}
			test_dataset = FireSequenceDataset(
				sample_indices=None,
				use_patches=use_eval_patches,
				**external_test_dataset_kwargs,
			)
	else:
		test_dataset = FireSequenceDataset(sample_indices=split_indices["test"], use_patches=use_eval_patches, **dataset_kwargs)

	train_dataloader_options = _resolve_dataloader_options(config, "train")
	val_dataloader_options = _resolve_dataloader_options(config, "val")
	test_dataloader_options = _resolve_dataloader_options(config, "test")
	train_loader = DataLoader(
		train_dataset,
		shuffle=True,
		**train_dataloader_options,
	)
	val_loader = DataLoader(
		val_dataset,
		shuffle=False,
		**val_dataloader_options,
	)
	test_loader = None
	if test_dataset is not None:
		test_loader = DataLoader(
			test_dataset,
			shuffle=False,
			**test_dataloader_options,
		)
	return train_loader, val_loader, test_loader


if __name__ == "__main__":
	from pathlib import Path

	from src.config import load_config

	if torch is None:
		print("smoke test skipped: PyTorch is not installed in this environment")
		raise SystemExit(0)

	project_root = Path(__file__).resolve().parents[2]
	config_path = project_root / "configs" / "default.yaml"
	config = load_config(config_path)
	config["config_path"] = str(config_path)

	train_loader, _, _ = create_dataloaders(config)
	x_batch, y_batch = next(iter(train_loader))[:2]
	assert x_batch.ndim == 5, f"Expected X batch to have 5 dimensions, got {tuple(x_batch.shape)}."
	assert y_batch.ndim == 4, f"Expected y batch to have 4 dimensions, got {tuple(y_batch.shape)}."
	print(f"smoke test passed: X={tuple(x_batch.shape)}, y={tuple(y_batch.shape)}")
