"""Pure target-construction functions for the rebuilt full-frame dataset."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.data.fire_mask_thresholds import threshold_union_mask


def build_processed_target(
	current_frame: np.ndarray,
	future_frame: np.ndarray,
	area_2d: np.ndarray,
	config: Mapping[str, Any],
	thresholds: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
	"""Build the v1 processed target from two raw-channel-first-or-last frames.

	Frames are accepted as H,W,C or C,H,W. Targets are always H,W and are never
	normalized here.
	"""
	current = _as_hwc(current_frame)
	future = _as_hwc(future_frame)
	if current.shape != future.shape or current.ndim != 3:
		raise ValueError(f"Current/future frames must have matching H,W,C shapes, got {current.shape} and {future.shape}")
	area = np.asarray(area_2d, dtype=np.float32)
	if area.shape != current.shape[:2]:
		raise ValueError(f"area_2d shape {area.shape} does not match frame shape {current.shape[:2]}")
	target_config = config.get("target_construction", {}) if isinstance(config.get("target_construction"), Mapping) else {}
	fuel_config = target_config.get("consumed_fuel", {}) if isinstance(target_config.get("consumed_fuel"), Mapping) else {}
	mask_config = target_config.get("fire_mask", {}) if isinstance(target_config.get("fire_mask"), Mapping) else {}
	energy_config = target_config.get("energy", {}) if isinstance(target_config.get("energy"), Mapping) else {}

	surface = current[:, :, 84] - future[:, :, 84]
	canopy = current[:, :, 85] - future[:, :, 85]
	if bool(fuel_config.get("clip_negative", True)):
		surface = np.maximum(surface, 0.0)
		canopy = np.maximum(canopy, 0.0)
	if bool(fuel_config.get("clip_to_available_fuel", True)):
		surface = np.minimum(surface, np.maximum(current[:, :, 84], 0.0))
		canopy = np.minimum(canopy, np.maximum(current[:, :, 85], 0.0))

	flux = future[:, :, 80] + future[:, :, 81] + future[:, :, 82] + future[:, :, 83]
	energy_mw = np.maximum(area * flux / 1.0e6, 0.0)
	energy_log = np.log1p(energy_mw).astype(np.float32)
	resolved_thresholds = thresholds or {
		"energy_threshold_mw": float(mask_config.get("energy_threshold_mw", 0.1)),
		"surface_fuel_threshold": float(mask_config.get("surface_fuel_threshold", mask_config.get("fuel_threshold", 0.001))),
		"canopy_fuel_threshold": float(mask_config.get("canopy_fuel_threshold", mask_config.get("fuel_threshold", 0.001))),
	}
	mask = threshold_union_mask(energy_mw, surface, canopy, resolved_thresholds)
	result = {
		"surface_consumed": np.asarray(surface, dtype=np.float32),
		"canopy_consumed": np.asarray(canopy, dtype=np.float32),
		"fire_mask": np.asarray(mask, dtype=bool),
		"energy_release_mw": np.asarray(energy_mw, dtype=np.float32),
		"energy_log": energy_log,
	}
	for name, value in result.items():
		if value.shape != current.shape[:2]:
			raise ValueError(f"Target {name} has shape {value.shape}, expected {current.shape[:2]}")
		if name != "fire_mask" and not np.isfinite(value).all():
			raise ValueError(f"Target {name} contains NaN or Inf")
	return result


def _as_hwc(frame: np.ndarray) -> np.ndarray:
	array = np.asarray(frame, dtype=np.float32)
	if array.ndim != 3:
		raise ValueError(f"Frame must be three-dimensional, got {array.shape}")
	# Processed frame files store raw channels as C,H,W. Prefer the
	# explicit 86-channel axis before checking the H,W,C form; dimensions
	# such as (86, 240, 144) otherwise look like a valid H,W,C tensor.
	if array.shape[0] == 86:
		return np.transpose(array, (1, 2, 0))
	if array.shape[-1] == 86:
		return array
	if array.shape[0] > 3 and array.shape[0] < array.shape[-1]:
		return np.transpose(array, (1, 2, 0))
	raise ValueError(f"Frame has no 86-channel dimension: {array.shape}")
