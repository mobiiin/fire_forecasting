"""PyTorch dataset and DataLoader helpers for sequence-to-map wildfire forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
	import torch  # type: ignore[import-not-found]
	from torch.utils.data import DataLoader, Dataset  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	DataLoader = None

	class Dataset:  # type: ignore[too-many-ancestors]
		"""Fallback base class used only when PyTorch is unavailable."""

		pass

from src.data.preprocessing import load_normalization_stats, normalize_channel_map, normalize_tensor
from src.data.splits import chronological_split_indices, chronological_train_val_split_indices


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

	numeric_suffixes = [_extract_numeric_suffix(path.stem) for path in file_paths]
	if all(value is not None for value in numeric_suffixes):
		return [path for _, path in sorted(zip(numeric_suffixes, file_paths), key=lambda item: item[0])]
	return sorted(file_paths, key=lambda path: path.name)


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


def _resolve_target_normalization_config(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Resolve target normalization configuration."""

	section = _get_section(config, "target_normalization")
	if not section:
		section = _get_section(config, "normalization")
	return {
		"enabled": bool(section.get("enabled", section.get("normalize_target", False))),
		"method": str(section.get("method", "zscore")).lower(),
	}


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
	total = 0
	if atmospheric["add_horizontal_wind_speed"]:
		total += num_vertical_levels
	if atmospheric["add_low_level_mean_wind_speed"]:
		total += 1
	if atmospheric["add_updraft"]:
		total += num_vertical_levels
	return total


def _count_engineered_channels(config: Mapping[str, Any]) -> int:
	"""Count the total number of engineered input channels that will be appended."""

	return _count_fuel_flux_engineered_channels(config) + count_atmospheric_engineered_channels(config)


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

	epsilon = max(float(atmospheric["epsilon"]), 0.0)
	per_frame_features: list[np.ndarray] = []
	u_levels: list[np.ndarray] = []
	v_levels: list[np.ndarray] = []

	for z_level in range(num_vertical_levels):
		base_index = z_level * variables_per_level
		u_values = np.asarray(raw_frame[:, :, base_index + 0], dtype=np.float32)
		v_values = np.asarray(raw_frame[:, :, base_index + 1], dtype=np.float32)
		w_values = np.asarray(raw_frame[:, :, base_index + 2], dtype=np.float32)
		u_levels.append(u_values)
		v_levels.append(v_values)
		if atmospheric["add_horizontal_wind_speed"]:
			wind_speed = np.sqrt(u_values * u_values + v_values * v_values + epsilon).astype(np.float32, copy=False)
			per_frame_features.append(wind_speed[:, :, None].astype(np.float32, copy=False))
		if atmospheric["add_updraft"]:
			updraft = np.maximum(w_values, 0.0)
			per_frame_features.append(None)  # placeholder to preserve deterministic order after low-level mean

	if atmospheric["add_low_level_mean_wind_speed"]:
		selected_u = np.stack([u_levels[index] for index in low_level_indices], axis=0)
		selected_v = np.stack([v_levels[index] for index in low_level_indices], axis=0)
		mean_low_u = np.mean(selected_u, axis=0)
		mean_low_v = np.mean(selected_v, axis=0)
		low_level_mean_wind_speed = np.sqrt(mean_low_u * mean_low_u + mean_low_v * mean_low_v + epsilon).astype(np.float32, copy=False)
		per_frame_features.append(low_level_mean_wind_speed[:, :, None].astype(np.float32, copy=False))

	if atmospheric["add_updraft"]:
		updraft_features: list[np.ndarray] = []
		for z_level in range(num_vertical_levels):
			base_index = z_level * variables_per_level
			w_values = np.asarray(raw_frame[:, :, base_index + 2], dtype=np.float32)
			updraft_features.append(np.maximum(w_values, 0.0)[:, :, None].astype(np.float32, copy=False))
		per_frame_features = [feature for feature in per_frame_features if feature is not None] + updraft_features

	if not per_frame_features:
		height, width = int(raw_frame.shape[0]), int(raw_frame.shape[1])
		return np.zeros((height, width, 0), dtype=np.float32)
	return np.concatenate(per_frame_features, axis=-1).astype(np.float32, copy=False)


def resolve_engineered_feature_slices(config: Mapping[str, Any], base_input_channel_count: int) -> dict[str, slice]:
	"""Resolve deterministic channel slices for all engineered feature groups."""

	offset = int(base_input_channel_count)
	atmospheric = _resolve_atmospheric_features_config(config)
	engineered = _resolve_engineered_features_config(config)
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
	return slices


def _load_initial_fuel_map(file_paths: Sequence[Path], config: Mapping[str, Any]) -> np.ndarray:
	"""Load the initial surface/canopy fuel map used for cumulative consumed-fuel features."""

	engineered = _resolve_engineered_features_config(config)
	if engineered["initial_fuel_mode"] != "first_dataset_frame":
		raise ValueError(
			f"Unsupported engineered_features.initial_fuel_mode: {engineered['initial_fuel_mode']!r}. "
			"Only 'first_dataset_frame' is currently supported."
		)
	layout = _resolve_channel_layout(config)
	first_frame = np.load(file_paths[0], mmap_mode="r", allow_pickle=False)
	return _slice_channels(first_frame, layout["fuel_channels"])


def build_engineered_features(
	input_frames: np.ndarray,
	file_paths: Sequence[str | Path],
	start_index: int,
	config: Mapping[str, Any],
) -> np.ndarray:
	"""Build leakage-safe engineered features from current and previous frames only."""

	engineered = _resolve_engineered_features_config(config)
	atmospheric_count = count_atmospheric_engineered_channels(config)
	fuel_flux_count = _count_fuel_flux_engineered_channels(config)
	if atmospheric_count + fuel_flux_count <= 0:
		height, width = int(input_frames.shape[1]), int(input_frames.shape[2])
		return np.zeros((int(input_frames.shape[0]), height, width, 0), dtype=np.float32)

	layout = _resolve_channel_layout(config)
	resolved_paths = [Path(path) for path in file_paths]
	raw_frames = np.asarray(input_frames, dtype=np.float32)
	if raw_frames.ndim != 4:
		raise ValueError(f"build_engineered_features expects raw input_frames shaped (T, H, W, C), got {raw_frames.shape}.")

	initial_fuel = None
	if engineered["enabled"] and engineered["add_cumulative_consumed_fuel"]:
		initial_fuel = _load_initial_fuel_map(resolved_paths, config)
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

		if per_timestep_features:
			feature_frames.append(np.concatenate(per_timestep_features, axis=-1).astype(np.float32, copy=False))
		else:
			height, width = current_frame.shape[:2]
			feature_frames.append(np.zeros((height, width, 0), dtype=np.float32))

	return np.stack(feature_frames, axis=0).astype(np.float32, copy=False)


def build_multitask_target(
	current_frame: np.ndarray,
	future_frame: np.ndarray,
	initial_fuel: np.ndarray,
	config: Mapping[str, Any],
) -> np.ndarray:
	"""Build the 3-channel multitask target."""

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

	target = np.stack(
		[
			surface_consumed_target.astype(np.float32, copy=False),
			canopy_consumed_target.astype(np.float32, copy=False),
			mask_array,
		],
		axis=0,
	)
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
		self.engineered_channel_count = self.fuel_flux_engineered_channel_count + self.atmospheric_engineered_channel_count
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
		self.target_mean, self.target_std = self._resolve_target_normalization_stats()
		self.initial_fuel_map = _load_initial_fuel_map(self.file_paths, self.config) if self.task_type == "multitask" or _resolve_engineered_features_config(self.config)["enabled"] else None

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

		if self.normalization_stats is None:
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
		if self.task_type == "multitask" and target_array.shape[0] != 3:
			raise ValueError(f"Expected multitask target shape (3, H, W), got {target_array.shape}.")

		x_tensor = torch.from_numpy(stacked_inputs).to(torch.float32)
		y_tensor = torch.from_numpy(target_array).to(torch.float32)
		if self.transform is not None:
			x_tensor = self.transform(x_tensor)
		if self.target_transform is not None:
			y_tensor = self.target_transform(y_tensor)

		if self.return_metadata:
			metadata = {
				"sample_index": sample_start,
				"current_index": current_index,
				"future_index": future_index,
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
			return x_tensor, y_tensor, metadata

		return x_tensor, y_tensor


def create_dataloaders(config):
	"""Build train/validation/test DataLoaders from a configuration dictionary."""

	if torch is None or DataLoader is None:
		raise ImportError("PyTorch is required to build DataLoaders for wildfire forecasting.")

	for required_key in ("data_dir", "file_pattern", "input_sequence_length", "prediction_horizon"):
		if required_key not in config:
			raise KeyError(f"Config is missing required key '{required_key}'.")

	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	data_dir = _resolve_path(config_path, config["data_dir"])
	file_pattern = str(config["file_pattern"])
	if not data_dir.exists():
		raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

	files = _sort_chronologically(list(data_dir.glob(file_pattern)))
	if not files:
		raise FileNotFoundError(f"No files found in '{data_dir}' using pattern '{file_pattern}'.")

	input_sequence_length = int(config["input_sequence_length"])
	prediction_horizon = int(config["prediction_horizon"])
	target_channel = int(config.get("target_channel", 0))
	input_channel_count = int(config.get("input_channel_count", config.get("model", {}).get("input_channels", 0)))
	if input_channel_count <= 0:
		raise KeyError("Config must define a positive input_channel_count or model.input_channels.")
	split_mode = str(config.get("split_mode", "train_val_test")).lower()
	train_fraction = float(config.get("train_fraction", 0.7))
	val_fraction = float(config.get("val_fraction", 0.15))
	test_fraction = float(config.get("test_fraction", 0.15))
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

	normalization_stats = None
	normalization_config = _get_section(config, "normalization")
	normalization_path = normalization_config.get("path")
	if normalization_path:
		resolved_normalization_path = _resolve_path(config_path, normalization_path)
		if resolved_normalization_path.exists():
			normalization_stats = resolved_normalization_path

	task_type = str(config.get("task_type", _get_section(config, "training").get("task_type", "regression"))).lower()
	target_normalization = _resolve_target_normalization_config(config)
	dataset_kwargs = {
		"file_paths": files,
		"input_sequence_length": input_sequence_length,
		"prediction_horizon": prediction_horizon,
		"target_channel": target_channel,
		"input_channel_count": input_channel_count,
		"input_channel_indices": config.get("input_channel_indices"),
		"task_type": task_type,
		"fire_threshold": float(config.get("fire_threshold", _get_section(config, "training").get("fire_threshold", 0.5))),
		"patch_size": int(config.get("patch_size", 64)),
		"active_patch_probability": float(config.get("active_patch_probability", 0.7)),
		"active_threshold": float(config.get("active_threshold", config.get("fire_threshold", 0.5))),
		"normalization_stats": normalization_stats,
		"normalize_target": bool(target_normalization["enabled"]),
		"config": config,
	}
	use_train_patches = bool(config.get("use_patches", False))
	use_eval_patches = bool(config.get("use_patches_for_eval", False))

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
			test_dataset = FireSequenceDataset(
				file_paths=external_test_files,
				sample_indices=None,
				use_patches=use_eval_patches,
				**dataset_kwargs,
			)
	else:
		test_dataset = FireSequenceDataset(sample_indices=split_indices["test"], use_patches=use_eval_patches, **dataset_kwargs)

	batch_size = int(config.get("batch_size", 4))
	num_workers = int(config.get("num_workers", 0))
	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
	)
	test_loader = None
	if test_dataset is not None:
		test_loader = DataLoader(
			test_dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=torch.cuda.is_available(),
			drop_last=False,
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
