from __future__ import annotations

import math

import pytest

from src.training.metrics import compute_metrics


def _multitask_energy_config() -> dict[str, object]:
	return {
		"task_type": "multitask",
		"energy_release": {
			"enabled": True,
			"target_transform": "log1p",
			"inverse_transform": "expm1",
			"predict_total": True,
			"predict_sensible": False,
			"predict_latent": False,
		},
		"multitask": {
			"energy_active_threshold_MW": 0.5,
			"consumed_active_threshold": 0.01,
		},
	}


def test_multitask_energy_log_mae_uses_direct_channel_three_error() -> None:
	torch = pytest.importorskip("torch")
	y_true = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
	y_pred = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
	y_true[:, 3] = torch.tensor([[[0.0, math.log1p(1.0)], [math.log1p(3.0), 0.0]]])
	y_pred[:, 3] = y_true[:, 3] + torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

	metrics = compute_metrics(y_pred, y_true, _multitask_energy_config())

	assert metrics["energy_log_mae"] == pytest.approx(2.5)
	assert "energy_mw_mae" in metrics
	assert "active_energy_mw_mae" in metrics


def test_multitask_active_energy_log_mae_uses_target_defined_active_pixels() -> None:
	torch = pytest.importorskip("torch")
	y_true = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
	y_pred = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
	y_true[:, 0] = torch.tensor([[[0.0, 0.0], [0.02, 0.0]]])
	y_true[:, 2] = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
	y_true[:, 3] = torch.tensor([[[0.0, math.log1p(1.0)], [0.0, 0.0]]])
	y_pred[:, 3] = y_true[:, 3] + torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

	metrics = compute_metrics(y_pred, y_true, _multitask_energy_config())

	assert metrics["active_energy_log_mae"] == pytest.approx(2.0)
