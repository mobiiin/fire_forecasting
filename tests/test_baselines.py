from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.baselines.common import build_mask_logits_from_binary, probability_to_logit
from src.baselines.linear_extrapolation import predict_linear_extrapolation_for_sample
from src.baselines.persistence import predict_persistence_for_sample
from src.training.metrics import compute_metrics


def _write_frame(path: Path, surface_flux: float, surface_fuel: float, canopy_fuel: float) -> None:
	frame = np.zeros((2, 2, 86), dtype=np.float32)
	frame[:, :, 80] = surface_flux
	frame[:, :, 81] = 0.0
	frame[:, :, 82] = 0.0
	frame[:, :, 83] = 0.0
	frame[:, :, 84] = surface_fuel
	frame[:, :, 85] = canopy_fuel
	np.save(path, frame)


def _config() -> dict[str, object]:
	return {
		"task_type": "multitask",
		"input_sequence_length": 2,
		"prediction_horizon": 1,
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
			"mask_target_type": "burned_fuel",
			"consumed_fuel_threshold": 0.01,
			"clamp_consumed_fuel_targets_nonnegative": True,
		},
		"energy_release": {
			"enabled": True,
			"surface_sensible_flux_channel": 80,
			"surface_latent_flux_channel": 81,
			"canopy_sensible_flux_channel": 82,
			"canopy_latent_flux_channel": 83,
			"target_transform": "log1p",
			"inverse_transform": "expm1",
			"predict_total": True,
			"predict_sensible": False,
			"predict_latent": False,
			"flux_units": "W_per_m2",
			"clamp_negative_flux_to_zero": True,
		},
		"baselines": {
			"persistence": {"mask_mode": "current_mask"},
			"linear_extrapolation": {"mask_source": "predicted_consumed", "consumed_threshold": 0.001},
		},
	}


def _dataset_record(tmp_path: Path) -> dict[str, object]:
	frame0 = tmp_path / "frame_000.npy"
	frame1 = tmp_path / "frame_001.npy"
	frame2 = tmp_path / "frame_002.npy"
	frame3 = tmp_path / "frame_003.npy"
	_write_frame(frame0, surface_flux=1.0e6, surface_fuel=10.0, canopy_fuel=5.0)
	_write_frame(frame1, surface_flux=2.0e6, surface_fuel=8.0, canopy_fuel=4.0)
	_write_frame(frame2, surface_flux=3.0e6, surface_fuel=6.0, canopy_fuel=3.0)
	_write_frame(frame3, surface_flux=4.0e6, surface_fuel=4.0, canopy_fuel=2.0)
	return {
		"dataset_id": 0,
		"dataset_name": "toy",
		"data_dir": tmp_path,
		"file_paths": [frame0, frame1, frame2, frame3],
		"raw_shape": (2, 2, 86),
		"geometry": {
			"area_2d_m2": np.ones((2, 2), dtype=np.float32),
			"geom_path": str(tmp_path / "toy.geom"),
			"terrain_path": None,
			"dy_m": 1.0,
			"dx_min_m": 1.0,
			"dx_max_m": 1.0,
			"dx_mean_m": 1.0,
			"area_min_m2": 1.0,
			"area_max_m2": 1.0,
			"area_mean_m2": 1.0,
		},
	}


def test_probability_to_logit_matches_expected_values() -> None:
	values = np.asarray([0.001, 0.5, 0.999], dtype=np.float32)
	logits = probability_to_logit(values)
	assert np.allclose(logits[1], 0.0, atol=1.0e-6)
	assert logits[0] < 0.0
	assert logits[2] > 0.0


def test_build_mask_logits_from_binary_has_active_and_inactive_states() -> None:
	mask = np.asarray([[0.0, 1.0]], dtype=np.float32)
	logits = build_mask_logits_from_binary(mask)
	assert logits.shape == mask.shape
	assert logits[0, 0] < 0.0
	assert logits[0, 1] > 0.0


def test_persistence_baseline_uses_zero_consumption_and_current_energy(tmp_path: Path) -> None:
	config = _config()
	record = _dataset_record(tmp_path)
	prediction = predict_persistence_for_sample(
		dataset_record=record,
		sample_ref={"sample_index": 0},
		config=config,
		patch=None,
	)
	assert prediction.shape == (4, 2, 2)
	assert np.allclose(prediction[0], 0.0)
	assert np.allclose(prediction[1], 0.0)
	assert np.all(prediction[2] > 0.0)
	expected_energy = np.log1p(np.full((2, 2), 2.0, dtype=np.float32))
	assert np.allclose(prediction[3], expected_energy)


def test_linear_extrapolation_scales_last_consumption_trend_and_energy(tmp_path: Path) -> None:
	config = _config()
	record = _dataset_record(tmp_path)
	prediction = predict_linear_extrapolation_for_sample(
		dataset_record=record,
		sample_ref={"sample_index": 0},
		config=config,
		patch={"y0": 0, "y1": 1, "x0": 0, "x1": 1},
	)
	assert prediction.shape == (4, 1, 1)
	assert np.allclose(prediction[0], 2.0)
	assert np.allclose(prediction[1], 1.0)
	assert prediction[2, 0, 0] > 0.0
	expected_energy = np.log1p(np.asarray([[3.0]], dtype=np.float32))
	assert np.allclose(prediction[3], expected_energy)


def test_linear_extrapolation_scales_by_prediction_horizon(tmp_path: Path) -> None:
	config = _config()
	config["prediction_horizon"] = 2
	record = _dataset_record(tmp_path)
	prediction = predict_linear_extrapolation_for_sample(
		dataset_record=record,
		sample_ref={"sample_index": 0},
		config=config,
		patch={"y0": 0, "y1": 1, "x0": 0, "x1": 1},
	)
	assert prediction.shape == (4, 1, 1)
	assert np.allclose(prediction[0], 4.0)
	assert np.allclose(prediction[1], 2.0)
	expected_energy = np.log1p(np.asarray([[4.0]], dtype=np.float32))
	assert np.allclose(prediction[3], expected_energy)


def test_baseline_predictions_produce_log_space_energy_metrics(tmp_path: Path) -> None:
	torch = pytest.importorskip("torch")
	config = _config()
	record = _dataset_record(tmp_path)
	target = np.zeros((4, 2, 2), dtype=np.float32)
	target[2] = 1.0
	target[3] = np.log1p(np.full((2, 2), 3.0, dtype=np.float32))

	for predict_fn in (predict_persistence_for_sample, predict_linear_extrapolation_for_sample):
		prediction = predict_fn(
			dataset_record=record,
			sample_ref={"sample_index": 0},
			config=config,
			patch=None,
		)
		metrics = compute_metrics(
			torch.from_numpy(prediction[None]).to(torch.float32),
			torch.from_numpy(target[None]).to(torch.float32),
			config,
		)

		assert "energy_log_mae" in metrics
		assert "active_energy_log_mae" in metrics
		assert np.isfinite(metrics["energy_log_mae"])
		assert np.isfinite(metrics["active_energy_log_mae"])
