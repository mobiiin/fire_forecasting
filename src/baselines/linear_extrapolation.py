"""Linear-extrapolation baseline for wildfire forecasting."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.baselines.common import (
	build_mask_logits_from_binary,
	compute_energy_target_channels_from_raw_frame,
	extract_raw_patch,
	get_raw_frames_for_sample,
	resolve_baseline_method_config,
	resolve_multitask_config,
)
from src.baselines.persistence import predict_persistence_for_sample
from src.data.energy_release import inverse_transform_energy_target, transform_energy_target


_WARNED_NO_PREVIOUS_FRAME = False


def _warn_no_previous_frame_once(sample_ref: Mapping[str, Any]) -> None:
	global _WARNED_NO_PREVIOUS_FRAME
	if _WARNED_NO_PREVIOUS_FRAME:
		return
	print(
		"WARNING: linear extrapolation fell back to persistence because no previous frame exists "
		f"for sample_index={int(sample_ref['sample_index'])}."
	)
	_WARNED_NO_PREVIOUS_FRAME = True


def predict_linear_extrapolation_for_sample(
	dataset_record: Mapping[str, Any],
	sample_ref: Mapping[str, Any],
	config: Mapping[str, Any],
	patch: Mapping[str, int] | None = None,
) -> np.ndarray:
	input_sequence_length = int(config["input_sequence_length"])
	prediction_horizon = int(config["prediction_horizon"])
	sample_index = int(sample_ref["sample_index"])
	frames = get_raw_frames_for_sample(
		dataset_record=dataset_record,
		sample_index=sample_index,
		input_sequence_length=input_sequence_length,
		prediction_horizon=prediction_horizon,
	)
	previous_frame = frames["previous_frame"]
	if previous_frame is None:
		_warn_no_previous_frame_once(sample_ref)
		return predict_persistence_for_sample(dataset_record=dataset_record, sample_ref=sample_ref, config=config, patch=patch)

	multitask = resolve_multitask_config(config)
	surface_channel = int(multitask["surface_fuel_channel"])
	canopy_channel = int(multitask["canopy_fuel_channel"])
	current_frame = np.asarray(frames["current_frame"], dtype=np.float32)
	previous_frame = np.asarray(previous_frame, dtype=np.float32)

	prev_surface = np.asarray(previous_frame[:, :, surface_channel], dtype=np.float32)
	curr_surface = np.asarray(current_frame[:, :, surface_channel], dtype=np.float32)
	prev_canopy = np.asarray(previous_frame[:, :, canopy_channel], dtype=np.float32)
	curr_canopy = np.asarray(current_frame[:, :, canopy_channel], dtype=np.float32)

	horizon = float(prediction_horizon)
	surface_pred = np.clip(horizon * np.maximum(prev_surface - curr_surface, 0.0), 0.0, curr_surface).astype(np.float32, copy=False)
	canopy_pred = np.clip(horizon * np.maximum(prev_canopy - curr_canopy, 0.0), 0.0, curr_canopy).astype(np.float32, copy=False)

	method_config = resolve_baseline_method_config(config, "linear_extrapolation")
	mask_source = str(method_config.get("mask_source", "predicted_consumed")).lower()
	consumed_threshold = float(method_config.get("consumed_threshold", 0.001))
	if mask_source != "predicted_consumed":
		raise ValueError(f"Unsupported baselines.linear_extrapolation.mask_source: {mask_source!r}.")
	mask_binary = (np.maximum(surface_pred, canopy_pred) > consumed_threshold).astype(np.float32, copy=False)
	mask_logits = build_mask_logits_from_binary(mask_binary)

	prev_energy_channels = compute_energy_target_channels_from_raw_frame(previous_frame, dataset_record, config)
	curr_energy_channels = compute_energy_target_channels_from_raw_frame(current_frame, dataset_record, config)
	energy_channels: list[np.ndarray] = []
	for prev_channel, curr_channel in zip(prev_energy_channels, curr_energy_channels):
		prev_energy = inverse_transform_energy_target(np.asarray(prev_channel, dtype=np.float32), config)
		curr_energy = inverse_transform_energy_target(np.asarray(curr_channel, dtype=np.float32), config)
		pred_energy = np.maximum(curr_energy + horizon * (curr_energy - prev_energy), 0.0).astype(np.float32, copy=False)
		energy_channels.append(transform_energy_target(pred_energy, config))

	prediction_channels = [
		extract_raw_patch(surface_pred, patch),
		extract_raw_patch(canopy_pred, patch),
		extract_raw_patch(mask_logits, patch),
		*[extract_raw_patch(channel, patch) for channel in energy_channels],
	]
	return np.stack(prediction_channels, axis=0).astype(np.float32, copy=False)
