"""Persistence baseline for wildfire forecasting."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.baselines.common import (
	build_current_mask,
	build_mask_logits_from_binary,
	compute_energy_target_channels_from_raw_frame,
	ensure_initial_fuel,
	extract_raw_patch,
	get_raw_frames_for_sample,
	resolve_baseline_method_config,
)


def predict_persistence_for_sample(
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
	current_frame = np.asarray(frames["current_frame"], dtype=np.float32)
	current_frame = extract_raw_patch(current_frame, patch)

	height, width = tuple(int(value) for value in current_frame.shape[:2])
	surface_pred = np.zeros((height, width), dtype=np.float32)
	canopy_pred = np.zeros((height, width), dtype=np.float32)

	method_config = resolve_baseline_method_config(config, "persistence")
	mask_mode = str(method_config.get("mask_mode", "zero")).lower()
	if mask_mode == "zero":
		mask_binary = np.zeros((height, width), dtype=np.float32)
	elif mask_mode == "current_mask":
		initial_fuel = ensure_initial_fuel(dataset_record, config)
		mask_binary = build_current_mask(
			current_frame=np.asarray(frames["current_frame"], dtype=np.float32),
			initial_fuel=initial_fuel,
			config=config,
		)
		mask_binary = extract_raw_patch(mask_binary, patch)
	else:
		raise ValueError(f"Unsupported baselines.persistence.mask_mode: {mask_mode!r}.")

	mask_logits = build_mask_logits_from_binary(mask_binary)
	energy_channels = [
		extract_raw_patch(channel, patch)
		for channel in compute_energy_target_channels_from_raw_frame(
			frame=np.asarray(frames["current_frame"], dtype=np.float32),
			dataset_record=dataset_record,
			config=config,
		)
	]

	prediction_channels = [surface_pred, canopy_pred, mask_logits, *energy_channels]
	return np.stack(prediction_channels, axis=0).astype(np.float32, copy=False)
