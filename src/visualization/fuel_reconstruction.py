"""Fuel-bed reconstruction helpers for multitask wildfire forecasts."""

from __future__ import annotations

import numpy as np

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None


def reconstruct_future_fuel_bed(
	current_surface_fuel,
	current_canopy_fuel,
	pred_surface_consumed,
	pred_canopy_consumed,
	clamp_nonnegative: bool = True,
):
	"""Reconstruct future surface/canopy fuel beds from consumed-fuel predictions."""

	if torch is not None and any(torch.is_tensor(value) for value in (
		current_surface_fuel,
		current_canopy_fuel,
		pred_surface_consumed,
		pred_canopy_consumed,
	)):
		if torch is None:
			raise ImportError("PyTorch is required for tensor-based fuel reconstruction.")
		current_surface = current_surface_fuel if torch.is_tensor(current_surface_fuel) else torch.as_tensor(current_surface_fuel)
		current_canopy = current_canopy_fuel if torch.is_tensor(current_canopy_fuel) else torch.as_tensor(current_canopy_fuel)
		surface_consumed = pred_surface_consumed if torch.is_tensor(pred_surface_consumed) else torch.as_tensor(pred_surface_consumed)
		canopy_consumed = pred_canopy_consumed if torch.is_tensor(pred_canopy_consumed) else torch.as_tensor(pred_canopy_consumed)
		pred_future_surface_fuel = current_surface - surface_consumed
		pred_future_canopy_fuel = current_canopy - canopy_consumed
		if clamp_nonnegative:
			pred_future_surface_fuel = torch.clamp(pred_future_surface_fuel, min=0.0)
			pred_future_canopy_fuel = torch.clamp(pred_future_canopy_fuel, min=0.0)
		return pred_future_surface_fuel, pred_future_canopy_fuel

	current_surface = np.asarray(current_surface_fuel, dtype=np.float32)
	current_canopy = np.asarray(current_canopy_fuel, dtype=np.float32)
	surface_consumed = np.asarray(pred_surface_consumed, dtype=np.float32)
	canopy_consumed = np.asarray(pred_canopy_consumed, dtype=np.float32)
	pred_future_surface_fuel = current_surface - surface_consumed
	pred_future_canopy_fuel = current_canopy - canopy_consumed
	if clamp_nonnegative:
		pred_future_surface_fuel = np.maximum(pred_future_surface_fuel, 0.0)
		pred_future_canopy_fuel = np.maximum(pred_future_canopy_fuel, 0.0)
	return pred_future_surface_fuel.astype(np.float32, copy=False), pred_future_canopy_fuel.astype(np.float32, copy=False)
