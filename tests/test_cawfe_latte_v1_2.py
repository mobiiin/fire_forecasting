"""CAWFE-Latte v1.2 temporal-attention-pooling ablation tests."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.models.cawfe_latte import CAWFELatte, CAWFELatteV11, CAWFELatteV12  # noqa: E402
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


def _config(architecture: str = "cawfe_latte_v1_2") -> dict:
	return {
		"task_type": "multitask",
		"model": {"architecture": architecture, "input_channels": CHANNELS, "output_channels": 4},
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
		"cawfe_latte_v1_2": {
			"enabled": True,
			"version": "v1_2_temporal_attention_pooling",
			"temporal_pooling": {
				"type": "attention",
				"enabled": True,
				"dim": 64,
				"hidden_dim": 16,
				"initialize_uniform": True,
			},
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


def test_model_factory_can_create_cawfe_latte_v1_2() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	assert isinstance(model, CAWFELatteV12)


def test_forward_returns_prediction_and_aux_with_terrain() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	out = model(_x(), terrain=_terrain())
	assert tuple(out["prediction"].shape) == (BATCH, 4, HEIGHT, WIDTH)
	assert tuple(out["aux_fire_support_logits"].shape) == (BATCH, 1, HEIGHT, WIDTH)


def test_temporal_attention_pooling_shape_sum_and_uniform_initialization() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	features = model(_x(), terrain=_terrain(), return_features=True)
	alpha = features["temporal_attention_alpha"]
	assert tuple(alpha.shape) == (BATCH, TIME, 1, HEIGHT, WIDTH)
	assert torch.allclose(alpha.sum(dim=1), torch.ones(BATCH, 1, HEIGHT, WIDTH), atol=1.0e-6)
	assert torch.allclose(alpha, torch.full_like(alpha, 1.0 / TIME), atol=1.0e-6)
	assert tuple(features["temporal_pooled_features"].shape) == (BATCH, 64, HEIGHT, WIDTH)
	assert tuple(features["temporal_attention_entropy"].shape) == (BATCH, 1, HEIGHT, WIDTH)


def test_support_gate_still_works_and_mask_logits_are_not_gated() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	features = model(_x(), terrain=_terrain(), return_features=True)
	assert float(features["support_gate"].detach().min()) >= 0.05 - 1.0e-6
	assert float(features["support_gate"].detach().max()) <= 1.0 + 1.0e-6
	assert torch.allclose(features["raw_mask_logits"], features["prediction"][:, 2:3])


def test_loss_accepts_v1_2_output_and_backward_works() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	criterion = MultiTaskLoss(_config())
	losses = criterion(model(_x(), terrain=_terrain()), _y())
	assert torch.isfinite(losses["total_loss"])
	assert "loss_aux_fire_support_total" in losses
	losses["total_loss"].backward()
	grad_sum = sum(float(param.grad.detach().abs().sum().item()) for param in model.parameters() if param.grad is not None)
	assert grad_sum > 0.0


def test_v1_and_v1_1_are_unaffected() -> None:
	v1 = CAWFELatte(input_channels=CHANNELS, input_sequence_length=TIME, output_channels=4)
	v11 = build_model_from_config(_config("cawfe_latte_v1_1"), input_channels=CHANNELS)
	assert isinstance(v11, CAWFELatteV11)
	assert not isinstance(v11, CAWFELatteV12)
	assert tuple(v1(_x(), return_features=True)["prediction"].shape) == (BATCH, 4, HEIGHT, WIDTH)
	v11_features = v11(_x(), terrain=_terrain(), return_features=True)
	assert "support_gate" in v11_features
	assert "temporal_attention_alpha" not in v11_features
