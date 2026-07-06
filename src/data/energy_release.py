"""Energy release map utilities derived from CAWFE flux channels."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _get_section(config: Mapping[str, Any] | None, *names: str) -> dict[str, Any]:
	"""Return the first nested mapping found under any of the provided names."""

	if not isinstance(config, Mapping):
		return {}
	for name in names:
		section = config.get(name)
		if isinstance(section, Mapping):
			return dict(section)
	return {}


def resolve_energy_release_config(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Resolve energy release configuration with defaults."""

	section = _get_section(config, "energy_release")
	for deprecated_key in ("dx_m", "dy_m", "fallback_dx_m", "fallback_dy_m"):
		if deprecated_key in section and section.get(deprecated_key) not in (None, "", "null"):
			print(
				f"WARNING: energy_release.{deprecated_key} is deprecated and ignored. "
				"Using per-cell area from .geom."
			)
	return {
		"enabled": bool(section.get("enabled", False)),
		"surface_sensible_flux_channel": int(section.get("surface_sensible_flux_channel", 80)),
		"surface_latent_flux_channel": int(section.get("surface_latent_flux_channel", 81)),
		"canopy_sensible_flux_channel": int(section.get("canopy_sensible_flux_channel", 82)),
		"canopy_latent_flux_channel": int(section.get("canopy_latent_flux_channel", 83)),
		"flux_units": str(section.get("flux_units", "W_per_m2")),
		"clamp_negative_flux_to_zero": bool(section.get("clamp_negative_flux_to_zero", True)),
		"save_components": bool(section.get("save_components", True)),
		"add_as_input_history": bool(section.get("add_as_input_history", False)),
		"target_transform": section.get("target_transform", "log1p"),
		"inverse_transform": section.get("inverse_transform", "expm1"),
		"predict_total": bool(section.get("predict_total", True)),
		"predict_sensible": bool(section.get("predict_sensible", False)),
		"predict_latent": bool(section.get("predict_latent", False)),
	}


def resolve_energy_output_channel_names(config: Mapping[str, Any]) -> list[str]:
	"""Return the enabled energy release target names in channel order."""

	energy = resolve_energy_release_config(config)
	if not energy["enabled"]:
		return []

	channel_names: list[str] = []
	if energy["predict_total"]:
		channel_names.append("energy_release_total_MW")
	if energy["predict_sensible"]:
		channel_names.append("energy_release_sensible_MW")
	if energy["predict_latent"]:
		channel_names.append("energy_release_latent_MW")
	return channel_names


def resolve_energy_target_count(config: Mapping[str, Any]) -> int:
	"""Return the number of enabled energy release targets."""

	return len(resolve_energy_output_channel_names(config))


def compute_energy_release_maps(
	frame: np.ndarray,
	config: Mapping[str, Any],
	area_2d_m2: np.ndarray,
) -> dict[str, np.ndarray]:
	"""Compute sensible, latent, and total energy release maps in MW per cell."""

	energy = resolve_energy_release_config(config)
	raw_frame = np.asarray(frame, dtype=np.float32)
	area_map = np.asarray(area_2d_m2, dtype=np.float32)
	if raw_frame.ndim != 3:
		raise ValueError(f"compute_energy_release_maps expects a frame shaped (H, W, C), got {raw_frame.shape}.")
	if area_map.shape != tuple(raw_frame.shape[:2]):
		raise ValueError(
			f"Area map shape from .geom does not match frame spatial shape. area={area_map.shape} frame={raw_frame.shape[:2]}"
		)
	if str(energy["flux_units"]) != "W_per_m2":
		raise ValueError(
			"Unsupported energy_release.flux_units. "
			f"Expected 'W_per_m2', got {energy['flux_units']!r}."
		)

	channel_indices = [
		int(energy["surface_sensible_flux_channel"]),
		int(energy["surface_latent_flux_channel"]),
		int(energy["canopy_sensible_flux_channel"]),
		int(energy["canopy_latent_flux_channel"]),
	]
	if max(channel_indices) >= int(raw_frame.shape[2]) or min(channel_indices) < 0:
		raise ValueError(
			f"Energy release flux channel indices {channel_indices} are out of bounds for frame with {raw_frame.shape[2]} channels."
		)

	surface_sensible = np.asarray(raw_frame[:, :, channel_indices[0]], dtype=np.float32)
	surface_latent = np.asarray(raw_frame[:, :, channel_indices[1]], dtype=np.float32)
	canopy_sensible = np.asarray(raw_frame[:, :, channel_indices[2]], dtype=np.float32)
	canopy_latent = np.asarray(raw_frame[:, :, channel_indices[3]], dtype=np.float32)

	if energy["clamp_negative_flux_to_zero"]:
		surface_sensible = np.maximum(surface_sensible, 0.0)
		surface_latent = np.maximum(surface_latent, 0.0)
		canopy_sensible = np.maximum(canopy_sensible, 0.0)
		canopy_latent = np.maximum(canopy_latent, 0.0)

	q_sensible_total = surface_sensible + canopy_sensible
	q_latent_total = surface_latent + canopy_latent
	q_total = q_sensible_total + q_latent_total

	energy_release_sensible_MW = (area_map * q_sensible_total / 1.0e6).astype(np.float32, copy=False)
	energy_release_latent_MW = (area_map * q_latent_total / 1.0e6).astype(np.float32, copy=False)
	energy_release_total_MW = (energy_release_sensible_MW + energy_release_latent_MW).astype(np.float32, copy=False)

	return {
		"energy_release_total_MW": energy_release_total_MW.astype(np.float32, copy=False),
		"energy_release_sensible_MW": energy_release_sensible_MW.astype(np.float32, copy=False),
		"energy_release_latent_MW": energy_release_latent_MW.astype(np.float32, copy=False),
		"q_sensible_total_W_m2": q_sensible_total.astype(np.float32, copy=False),
		"q_latent_total_W_m2": q_latent_total.astype(np.float32, copy=False),
		"q_total_W_m2": q_total.astype(np.float32, copy=False),
	}


def transform_energy_target(energy_MW: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
	"""Transform physical MW targets into the training space."""

	energy = resolve_energy_release_config(config)
	target_transform = energy["target_transform"]
	values = np.asarray(energy_MW, dtype=np.float32)
	if target_transform in ("log1p",):
		return np.log1p(np.maximum(values, 0.0)).astype(np.float32, copy=False)
	if target_transform in ("none", None, "null"):
		return values.astype(np.float32, copy=False)
	raise ValueError(
		"Unsupported energy_release.target_transform. "
		f"Expected 'log1p' or 'none', got {target_transform!r}."
	)


def inverse_transform_energy_target(y: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
	"""Invert the training-space energy target into physical MW."""

	energy = resolve_energy_release_config(config)
	target_transform = energy["target_transform"]
	inverse_transform = energy["inverse_transform"]
	values = np.asarray(y, dtype=np.float32)
	if inverse_transform == "expm1" or target_transform == "log1p":
		return np.expm1(values).astype(np.float32, copy=False)
	if target_transform in ("none", None, "null") or inverse_transform in ("none", None, "null"):
		return values.astype(np.float32, copy=False)
	raise ValueError(
		"Unsupported energy_release.inverse_transform. "
		f"Expected 'expm1' or 'none', got {inverse_transform!r}."
	)
