from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.dataset import FireSequenceDataset

pytest.importorskip("torch")


def _write_frame(path: Path, index: int) -> None:
	frame = np.zeros((3, 3, 86), dtype=np.float32)
	frame[..., 80] = 1.0 if index == 14 else 0.0
	frame[..., 84] = 100.0 - float(index)
	frame[..., 85] = 50.0 - 0.5 * float(index)
	np.save(path, frame)


def _config() -> dict:
	return {
		"input_sequence_length": 5,
		"prediction_horizon": 10,
		"input_channel_count": 86,
		"task_type": "multitask",
		"channel_layout": {
			"flux_channels": [80, 81, 82, 83],
			"fuel_channels": [84, 85],
			"surface_fuel_channel": 84,
			"canopy_fuel_channel": 85,
			"flux_mask_channel": 80,
		},
		"multitask": {
			"surface_fuel_channel": 84,
			"canopy_fuel_channel": 85,
			"flux_mask_channel": 80,
			"mask_target_type": "active_flux",
			"flux_fire_threshold": 0.5,
			"consumed_fuel_threshold": 0.01,
			"clamp_consumed_fuel_targets_nonnegative": True,
		},
		"engineered_features": {"enabled": False},
		"atmospheric_features": {"enabled": False},
		"energy_release": {"enabled": False},
	}


def test_t5_h10_sample_targets_start_plus_14(tmp_path: Path) -> None:
	paths = []
	for index in range(15):
		path = tmp_path / f"frame_{index:04d}.npy"
		_write_frame(path, index)
		paths.append(path)

	dataset = FireSequenceDataset(
		file_paths=paths,
		sample_indices=None,
		input_sequence_length=5,
		prediction_horizon=10,
		target_channel=84,
		input_channel_count=86,
		task_type="multitask",
		return_metadata=True,
		config=_config(),
	)

	assert len(dataset) == 1
	x, y, metadata = dataset[0]

	assert tuple(x.shape) == (5, 86, 3, 3)
	assert tuple(y.shape) == (3, 3, 3)
	assert metadata["start_idx"] == 0
	assert metadata["input_indices"] == [0, 1, 2, 3, 4]
	assert metadata["last_input_idx"] == 4
	assert metadata["target_idx"] == 14
	assert metadata["prediction_horizon"] == 10
	assert metadata["input_sequence_length"] == 5
	assert metadata["target_offset_from_start"] == 14
	assert metadata["target_offset_from_last_input"] == 10
	assert np.allclose(y[0].numpy(), 10.0)
	assert np.allclose(y[1].numpy(), 5.0)
	assert np.allclose(y[2].numpy(), 1.0)
