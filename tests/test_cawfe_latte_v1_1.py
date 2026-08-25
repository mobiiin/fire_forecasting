"""CAWFE-Latte v1.1 ablation tests."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.models.cawfe_latte import CAWFELatte, CAWFELatteV11  # noqa: E402
from src.models.model_factory import build_model_from_config  # noqa: E402
from src.training.losses import MultiTaskLoss  # noqa: E402


BATCH = 2
TIME = 5
CHANNELS = 129
HEIGHT = 8
WIDTH = 8


def _x() -> torch.Tensor:
	return torch.randn(BATCH, TIME, CHANNELS, HEIGHT, WIDTH)


def _terrain() -> torch.Tensor:
	return torch.rand(BATCH, 4, HEIGHT, WIDTH)


def _y() -> torch.Tensor:
	y = torch.randn(BATCH, 4, HEIGHT, WIDTH) * 0.1
	y[:, 2:3] = (torch.rand(BATCH, 1, HEIGHT, WIDTH) > 0.7).float()
	y[:, 3:4] = torch.rand(BATCH, 1, HEIGHT, WIDTH)
	return y


def _config() -> dict:
	return {
		"task_type": "multitask",
		"model": {"architecture": "cawfe_latte_v1_1", "input_channels": CHANNELS, "output_channels": 4},
		"input_sequence_length": TIME,
		"dataloader": {"source": "processed_full_frames"},
		"cawfe_latte": {
			"input_channels": CHANNELS,
			"input_sequence_length": TIME,
			"output_channels": 4,
			"output_dim": 64,
			"use_terrain_conditioning": True,
		},
		"cawfe_latte_v1_1": {
			"enabled": True,
			"version": "v1_1_resblocks_support_gate",
			"post_fusion_backbone": {"type": "residual_spatiotemporal", "dim": 64, "num_blocks": 6},
			"support_gate": {"enabled": True, "gate_min": 0.05, "gate_max": 1.0},
			"auxiliary": {"fire_support_head": {"enabled": True, "source": "support_logits", "weight": 0.2}},
		},
		"energy_release": {"enabled": True, "output_mode": "total", "target_transform": "log1p"},
		"training": {
			"task_type": "multitask",
			"loss": {
				"surface": {"type": "huber", "weight": 1.0, "delta": 1.0},
				"canopy": {"type": "huber", "weight": 1.0, "delta": 1.0},
				"mask": {"type": "bce_dice", "weight": 5.0, "bce_weight": 1.0, "dice_weight": 1.0},
				"energy": {"type": "huber", "weight": 1.0, "delta": 1.0},
				"auxiliary_fire_support": {"enabled": True, "weight": 0.2},
			},
		},
	}


def test_model_factory_can_create_cawfe_latte_v1_1() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	assert isinstance(model, CAWFELatteV11)


def test_forward_returns_prediction_and_aux_with_terrain() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	out = model(_x(), terrain=_terrain())
	assert tuple(out["prediction"].shape) == (BATCH, 4, HEIGHT, WIDTH)
	assert tuple(out["aux_fire_support_logits"].shape) == (BATCH, 1, HEIGHT, WIDTH)


def test_support_gate_range_and_feature_shapes() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	features = model(_x(), terrain=_terrain(), return_features=True)
	assert tuple(features["post_fusion_features"].shape) == (BATCH, TIME, 64, HEIGHT, WIDTH)
	assert tuple(features["decoded"].shape) == (BATCH, 64, HEIGHT, WIDTH)
	assert tuple(features["support_gate"].shape) == (BATCH, 1, HEIGHT, WIDTH)
	assert float(features["support_gate"].detach().min()) >= 0.05 - 1.0e-6
	assert float(features["support_gate"].detach().max()) <= 1.0 + 1.0e-6


def test_mask_logits_are_not_gated() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	features = model(_x(), terrain=_terrain(), return_features=True)
	assert torch.allclose(features["raw_mask_logits"], features["prediction"][:, 2:3])


def test_regression_channels_are_gated() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	features = model(_x(), terrain=_terrain(), return_features=True)
	prediction = features["prediction"]
	assert not torch.allclose(features["raw_surface"], prediction[:, 0:1])
	assert not torch.allclose(features["raw_canopy"], prediction[:, 1:2])
	assert not torch.allclose(features["raw_energy"], prediction[:, 3:4])


def test_missing_terrain_raises_when_conditioning_enabled() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	with pytest.raises(ValueError, match="terrain conditioning"):
		model(_x())


def test_loss_accepts_v1_1_output_and_backward_works() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	criterion = MultiTaskLoss(_config())
	losses = criterion(model(_x(), terrain=_terrain()), _y())
	assert torch.isfinite(losses["total_loss"])
	assert "loss_aux_fire_support_total" in losses
	assert "weighted_aux_fire_support" in losses
	losses["total_loss"].backward()
	grad_sum = sum(float(param.grad.detach().abs().sum().item()) for param in model.parameters() if param.grad is not None)
	assert grad_sum > 0.0


def test_v1_is_unaffected_and_has_no_v1_1_support_gate_features() -> None:
	model = CAWFELatte(input_channels=CHANNELS, input_sequence_length=TIME, output_channels=4)
	features = model(_x(), return_features=True)
	assert isinstance(model, CAWFELatte)
	assert not isinstance(model, CAWFELatteV11)
	assert "support_gate" not in features
	assert tuple(features["prediction"].shape) == (BATCH, 4, HEIGHT, WIDTH)
