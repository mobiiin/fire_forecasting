"""Spatial patch utilities for wildfire sequence datasets."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


Patch = dict[str, int]


def resolve_patching_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
	"""Resolve patching settings with backward-compatible fallbacks."""

	section = config.get("patching", {}) if isinstance(config, Mapping) and isinstance(config.get("patching"), Mapping) else {}
	patch_size = int(section.get("patch_size", config.get("patch_size", 64) if isinstance(config, Mapping) else 64))
	patch_height = int(section.get("patch_height", patch_size))
	patch_width = int(section.get("patch_width", patch_size))
	return {
		"enabled": bool(section.get("enabled", config.get("use_patches", False) if isinstance(config, Mapping) else False)),
		"patch_size": int(patch_size),
		"patch_height": int(patch_height),
		"patch_width": int(patch_width),
		"train_sampling_mode": str(section.get("train_sampling_mode", "mixed_active_random")).lower(),
		"active_patch_probability": float(section.get("active_patch_probability", config.get("active_patch_probability", 0.7) if isinstance(config, Mapping) else 0.7)),
		"random_patch_probability": float(section.get("random_patch_probability", 0.3)),
		"active_source": str(section.get("active_source", "combined_target")).lower(),
		"consumed_active_threshold": float(section.get("consumed_active_threshold", config.get("active_threshold", 0.001) if isinstance(config, Mapping) else 0.001)),
		"energy_active_threshold_MW": float(section.get("energy_active_threshold_MW", 0.001)),
		"min_active_pixels": int(section.get("min_active_pixels", 1)),
		"center_on_active_pixel": bool(section.get("center_on_active_pixel", True)),
		"jitter_active_center": bool(section.get("jitter_active_center", True)),
		"max_center_jitter_pixels": int(section.get("max_center_jitter_pixels", 16)),
		"eval_mode": str(section.get("eval_mode", "sliding_window")).lower(),
		"eval_patch_size": int(section.get("eval_patch_size", patch_size)),
		"eval_stride": int(section.get("eval_stride", max(1, patch_size // 2))),
		"include_border_patches": bool(section.get("include_border_patches", True)),
		"allow_padding_small_domains": bool(section.get("allow_padding_small_domains", False)),
		"pad_value_normalized": float(section.get("pad_value_normalized", 0.0)),
		"require_patch_divisible_by": int(section.get("require_patch_divisible_by", 1)),
		"auto_pad_to_divisible": bool(section.get("auto_pad_to_divisible", False)),
		"eval_use_patches": bool(section.get("enabled", config.get("use_patches_for_eval", False) if isinstance(config, Mapping) else False)),
	}


def validate_patch_dict(patch: Mapping[str, int]) -> Patch:
	"""Validate a patch dictionary and normalize integer fields."""

	required = ("y0", "y1", "x0", "x1")
	missing = [key for key in required if key not in patch]
	if missing:
		raise KeyError(f"Patch is missing required key(s): {', '.join(missing)}.")
	normalized = {key: int(patch[key]) for key in required}
	if normalized["y1"] <= normalized["y0"] or normalized["x1"] <= normalized["x0"]:
		raise ValueError(f"Patch bounds must be strictly increasing, got {normalized}.")
	return normalized


def make_patch(y0: int, y1: int, x0: int, x1: int) -> Patch:
	"""Construct a normalized patch dictionary."""

	return validate_patch_dict({"y0": int(y0), "y1": int(y1), "x0": int(x0), "x1": int(x1)})


def get_valid_patch_bounds(height: int, width: int, patch_h: int, patch_w: int) -> list[Patch]:
	"""Return every valid top-left-aligned patch of fixed size in a grid."""

	if height <= 0 or width <= 0:
		raise ValueError(f"Spatial dimensions must be positive, got H={height}, W={width}.")
	if patch_h <= 0 or patch_w <= 0:
		raise ValueError(f"Patch dimensions must be positive, got patch_h={patch_h}, patch_w={patch_w}.")
	if patch_h > height or patch_w > width:
		return []
	patches: list[Patch] = []
	for y0 in range(0, height - patch_h + 1):
		for x0 in range(0, width - patch_w + 1):
			patches.append(make_patch(y0=y0, y1=y0 + patch_h, x0=x0, x1=x0 + patch_w))
	return patches


def sample_random_patch(height: int, width: int, patch_h: int, patch_w: int, rng: np.random.Generator) -> Patch:
	"""Sample one uniformly random valid patch."""

	if patch_h > height or patch_w > width:
		raise ValueError(
			f"Patch dimensions must fit inside the array. Got H={height}, W={width}, patch_h={patch_h}, patch_w={patch_w}."
		)
	max_y0 = height - patch_h
	max_x0 = width - patch_w
	y0 = int(rng.integers(0, max_y0 + 1)) if max_y0 > 0 else 0
	x0 = int(rng.integers(0, max_x0 + 1)) if max_x0 > 0 else 0
	return make_patch(y0=y0, y1=y0 + patch_h, x0=x0, x1=x0 + patch_w)


def _clip_patch_origin(origin: int, patch_size: int, axis_size: int) -> int:
	"""Clip one patch origin to remain within an axis."""

	return int(max(0, min(origin, axis_size - patch_size)))


def sample_active_patch(
	activity_map: np.ndarray,
	patch_h: int,
	patch_w: int,
	rng: np.random.Generator,
	min_active_pixels: int,
	center_on_active_pixel: bool,
	jitter_active_center: bool,
	max_center_jitter_pixels: int,
) -> Patch | None:
	"""Sample a patch centered near active pixels when enough activity exists."""

	activity = np.asarray(activity_map)
	if activity.ndim != 2:
		raise ValueError(f"activity_map must be 2D, got shape {activity.shape}.")
	height, width = int(activity.shape[0]), int(activity.shape[1])
	if patch_h > height or patch_w > width:
		return None
	active_pixels = np.argwhere(activity > 0)
	if active_pixels.shape[0] < int(min_active_pixels):
		return None
	if not center_on_active_pixel:
		selected = active_pixels[rng.integers(active_pixels.shape[0])]
		return make_patch(
			y0=_clip_patch_origin(int(selected[0]), patch_h, height),
			y1=_clip_patch_origin(int(selected[0]), patch_h, height) + patch_h,
			x0=_clip_patch_origin(int(selected[1]), patch_w, width),
			x1=_clip_patch_origin(int(selected[1]), patch_w, width) + patch_w,
		)

	center_y, center_x = active_pixels[rng.integers(active_pixels.shape[0])]
	center_y = int(center_y)
	center_x = int(center_x)
	if jitter_active_center and max_center_jitter_pixels > 0:
		center_y += int(rng.integers(-max_center_jitter_pixels, max_center_jitter_pixels + 1))
		center_x += int(rng.integers(-max_center_jitter_pixels, max_center_jitter_pixels + 1))
	y0 = _clip_patch_origin(center_y - patch_h // 2, patch_h, height)
	x0 = _clip_patch_origin(center_x - patch_w // 2, patch_w, width)
	return make_patch(y0=y0, y1=y0 + patch_h, x0=x0, x1=x0 + patch_w)


def extract_patch_array(array: np.ndarray, patch: Mapping[str, int]) -> np.ndarray:
	"""Extract a spatial patch from an array whose last two spatial dims are H/W or H/W/C."""

	validated = validate_patch_dict(patch)
	array = np.asarray(array)
	if array.ndim == 2:
		return array[validated["y0"] : validated["y1"], validated["x0"] : validated["x1"]]
	if array.ndim == 3:
		return array[validated["y0"] : validated["y1"], validated["x0"] : validated["x1"], ...]
	if array.ndim == 4:
		return array[:, validated["y0"] : validated["y1"], validated["x0"] : validated["x1"], ...]
	raise ValueError(f"extract_patch_array expects 2D, 3D, or 4D input, got shape {array.shape}.")


def _positions_for_axis(axis_size: int, patch_size: int, stride: int, include_border_patches: bool) -> list[int]:
	"""Return start positions along one spatial axis."""

	if patch_size > axis_size:
		return []
	if stride <= 0:
		raise ValueError(f"stride must be positive, got {stride}.")
	positions = list(range(0, max(axis_size - patch_size, 0) + 1, stride))
	if not positions:
		positions = [0]
	last = axis_size - patch_size
	if include_border_patches and positions[-1] != last:
		positions.append(last)
	return sorted(set(int(pos) for pos in positions))


def build_sliding_window_patches(
	height: int,
	width: int,
	patch_h: int,
	patch_w: int,
	stride_h: int,
	stride_w: int,
	include_border_patches: bool,
) -> list[Patch]:
	"""Build deterministic sliding-window patches covering a full domain."""

	y_positions = _positions_for_axis(height, patch_h, stride_h, include_border_patches)
	x_positions = _positions_for_axis(width, patch_w, stride_w, include_border_patches)
	patches: list[Patch] = []
	for y0 in y_positions:
		for x0 in x_positions:
			patches.append(make_patch(y0=y0, y1=y0 + patch_h, x0=x0, x1=x0 + patch_w))
	return patches


def describe_patch_grid(height: int, width: int, patches: Sequence[Mapping[str, int]]) -> dict[str, int]:
	"""Summarize a patch grid for diagnostics."""

	patch_list = [validate_patch_dict(patch) for patch in patches]
	return {
		"height": int(height),
		"width": int(width),
		"num_patches": int(len(patch_list)),
	}
