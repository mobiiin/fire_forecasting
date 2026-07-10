"""Shared helpers for non-neural wildfire baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.energy_release import compute_energy_release_maps, resolve_energy_output_channel_names, transform_energy_target
from src.data.geometry import load_fire_geometry
from src.data.patching import extract_patch_array, validate_patch_dict


def _get_section(config: Mapping[str, Any] | None, *names: str) -> dict[str, Any]:
	if not isinstance(config, Mapping):
		return {}
	for name in names:
		section = config.get(name)
		if isinstance(section, Mapping):
			return dict(section)
	return {}


def resolve_channel_layout(config: Mapping[str, Any]) -> dict[str, int]:
	layout = _get_section(config, "channel_layout")
	if not layout:
		raise KeyError("Config is missing channel_layout.")
	fuel_channels = layout.get("fuel_channels")
	flux_channels = layout.get("flux_channels")
	if not isinstance(fuel_channels, Sequence) or len(fuel_channels) < 2:
		raise ValueError("channel_layout.fuel_channels must contain at least surface and canopy channels.")
	if not isinstance(flux_channels, Sequence) or not flux_channels:
		raise ValueError("channel_layout.flux_channels must be configured.")
	return {
		"surface_fuel_channel": int(layout.get("surface_fuel_channel", fuel_channels[0])),
		"canopy_fuel_channel": int(layout.get("canopy_fuel_channel", fuel_channels[1])),
		"flux_mask_channel": int(layout.get("flux_mask_channel", flux_channels[0])),
	}


def resolve_multitask_config(config: Mapping[str, Any]) -> dict[str, Any]:
	multitask = _get_section(config, "multitask")
	layout = resolve_channel_layout(config)
	return {
		"surface_fuel_channel": int(multitask.get("surface_fuel_channel", layout["surface_fuel_channel"])),
		"canopy_fuel_channel": int(multitask.get("canopy_fuel_channel", layout["canopy_fuel_channel"])),
		"flux_mask_channel": int(multitask.get("flux_mask_channel", layout["flux_mask_channel"])),
		"mask_target_type": str(multitask.get("mask_target_type", "active_flux")).lower(),
		"flux_fire_threshold": float(multitask.get("flux_fire_threshold", 0.05)),
		"consumed_fuel_threshold": float(multitask.get("consumed_fuel_threshold", 0.01)),
		"clamp_consumed_fuel_targets_nonnegative": bool(multitask.get("clamp_consumed_fuel_targets_nonnegative", True)),
	}


def resolve_baseline_method_config(config: Mapping[str, Any], method_name: str) -> dict[str, Any]:
	baselines = _get_section(config, "baselines")
	section = baselines.get(method_name)
	return dict(section) if isinstance(section, Mapping) else {}


def probability_to_logit(prob: np.ndarray | float, eps: float = 1.0e-6) -> np.ndarray:
	values = np.asarray(prob, dtype=np.float32)
	clipped = np.clip(values, eps, 1.0 - eps)
	return np.log(clipped / (1.0 - clipped)).astype(np.float32, copy=False)


def build_mask_logits_from_binary(
	mask: np.ndarray,
	active_prob: float = 0.999,
	inactive_prob: float = 0.001,
) -> np.ndarray:
	mask_array = np.asarray(mask, dtype=np.float32)
	probabilities = np.where(mask_array > 0.5, float(active_prob), float(inactive_prob)).astype(np.float32, copy=False)
	return probability_to_logit(probabilities)


def resolve_patch(patch: Mapping[str, Any] | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, int] | None:
	if isinstance(patch, Mapping):
		return validate_patch_dict(patch)
	if not isinstance(metadata, Mapping):
		return None
	if isinstance(metadata.get("patch"), Mapping):
		return validate_patch_dict(metadata["patch"])
	required = ("patch_top", "patch_left", "patch_bottom", "patch_right")
	if all(metadata.get(key) is not None for key in required):
		return validate_patch_dict(
			{
				"y0": int(metadata["patch_top"]),
				"y1": int(metadata["patch_bottom"]),
				"x0": int(metadata["patch_left"]),
				"x1": int(metadata["patch_right"]),
			}
		)
	if metadata.get("patch_top") is not None and metadata.get("patch_left") is not None and metadata.get("patch_size") is not None:
		y0 = int(metadata["patch_top"])
		x0 = int(metadata["patch_left"])
		size = int(metadata["patch_size"])
		return validate_patch_dict({"y0": y0, "y1": y0 + size, "x0": x0, "x1": x0 + size})
	return None


def extract_raw_patch(frame: np.ndarray, patch: Mapping[str, int] | None) -> np.ndarray:
	array = np.asarray(frame, dtype=np.float32)
	if patch is None:
		return array
	return np.asarray(extract_patch_array(array, validate_patch_dict(patch)), dtype=np.float32)


def ensure_geometry(dataset_record: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
	existing = dataset_record.get("geometry")
	if isinstance(existing, Mapping) and existing:
		return dict(existing)
	height, width = tuple(int(value) for value in dataset_record["raw_shape"][:2])
	return load_fire_geometry(
		data_dir=Path(dataset_record["data_dir"]).expanduser().resolve(),
		config=config,
		geom_path=dataset_record.get("geom_path"),
		terrain_path=dataset_record.get("terrain_path"),
		expected_shape=(height, width),
	)


def ensure_initial_fuel(dataset_record: Mapping[str, Any], config: Mapping[str, Any]) -> np.ndarray:
	cached = dataset_record.get("initial_fuel")
	if cached is not None:
		return np.asarray(cached, dtype=np.float32)
	file_paths = dataset_record.get("file_paths")
	if not isinstance(file_paths, Sequence) or not file_paths:
		raise ValueError("dataset_record.file_paths is required to load initial fuel.")
	layout = resolve_channel_layout(config)
	first_frame = np.load(Path(file_paths[0]).expanduser().resolve(), mmap_mode="r", allow_pickle=False)
	initial_fuel = np.stack(
		[
			np.asarray(first_frame[:, :, layout["surface_fuel_channel"]], dtype=np.float32),
			np.asarray(first_frame[:, :, layout["canopy_fuel_channel"]], dtype=np.float32),
		],
		axis=-1,
	).astype(np.float32, copy=False)
	return initial_fuel


def compute_energy_log_from_raw_frame(frame: np.ndarray, dataset_record: Mapping[str, Any], config: Mapping[str, Any]) -> np.ndarray:
	geometry = ensure_geometry(dataset_record, config)
	energy_maps = compute_energy_release_maps(
		frame=np.asarray(frame, dtype=np.float32),
		config=config,
		area_2d_m2=np.asarray(geometry["area_2d_m2"], dtype=np.float32),
	)
	return transform_energy_target(energy_maps["energy_release_total_MW"], config)


def compute_energy_target_channels_from_raw_frame(
	frame: np.ndarray,
	dataset_record: Mapping[str, Any],
	config: Mapping[str, Any],
) -> list[np.ndarray]:
	geometry = ensure_geometry(dataset_record, config)
	energy_maps = compute_energy_release_maps(
		frame=np.asarray(frame, dtype=np.float32),
		config=config,
		area_2d_m2=np.asarray(geometry["area_2d_m2"], dtype=np.float32),
	)
	output_names = resolve_energy_output_channel_names(config)
	return [transform_energy_target(energy_maps[name], config) for name in output_names]


def get_raw_frames_for_sample(
	dataset_record: Mapping[str, Any],
	sample_index: int,
	input_sequence_length: int,
	prediction_horizon: int,
) -> dict[str, Any]:
	file_paths = dataset_record.get("file_paths")
	if not isinstance(file_paths, Sequence) or not file_paths:
		raise ValueError("dataset_record.file_paths must be a non-empty sequence.")
	current_idx = int(sample_index) + int(input_sequence_length) - 1
	future_idx = current_idx + int(prediction_horizon)
	previous_idx = current_idx - 1
	if current_idx < 0 or current_idx >= len(file_paths):
		raise IndexError(f"current_idx={current_idx} is out of bounds for dataset {dataset_record.get('dataset_name', 'unknown')}.")
	if future_idx < 0 or future_idx >= len(file_paths):
		raise IndexError(f"future_idx={future_idx} is out of bounds for dataset {dataset_record.get('dataset_name', 'unknown')}.")

	def _load(index: int) -> np.ndarray:
		return np.asarray(np.load(Path(file_paths[index]).expanduser().resolve(), mmap_mode="r", allow_pickle=False), dtype=np.float32)

	previous_frame = _load(previous_idx) if previous_idx >= 0 else None
	current_frame = _load(current_idx)
	future_frame = _load(future_idx)
	return {
		"previous_idx": int(previous_idx),
		"current_idx": int(current_idx),
		"future_idx": int(future_idx),
		"previous_frame": previous_frame,
		"current_frame": current_frame,
		"future_frame": future_frame,
	}


def build_current_mask(
	current_frame: np.ndarray,
	initial_fuel: np.ndarray,
	config: Mapping[str, Any],
) -> np.ndarray:
	multitask = resolve_multitask_config(config)
	mask_target_type = str(multitask["mask_target_type"]).lower()
	if mask_target_type == "active_flux":
		flux = np.asarray(current_frame[:, :, int(multitask["flux_mask_channel"])], dtype=np.float32)
		return (flux > float(multitask["flux_fire_threshold"])).astype(np.float32, copy=False)
	if mask_target_type == "burned_fuel":
		surface_channel = int(multitask["surface_fuel_channel"])
		canopy_channel = int(multitask["canopy_fuel_channel"])
		surface_consumed = np.asarray(initial_fuel[:, :, 0], dtype=np.float32) - np.asarray(current_frame[:, :, surface_channel], dtype=np.float32)
		canopy_consumed = np.asarray(initial_fuel[:, :, 1], dtype=np.float32) - np.asarray(current_frame[:, :, canopy_channel], dtype=np.float32)
		if bool(multitask["clamp_consumed_fuel_targets_nonnegative"]):
			surface_consumed = np.maximum(surface_consumed, 0.0)
			canopy_consumed = np.maximum(canopy_consumed, 0.0)
		return (np.maximum(surface_consumed, canopy_consumed) > float(multitask["consumed_fuel_threshold"])).astype(np.float32, copy=False)
	raise ValueError(f"Unsupported multitask.mask_target_type for current-mask baseline: {mask_target_type!r}.")
