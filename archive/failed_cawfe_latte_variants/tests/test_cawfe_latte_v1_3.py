"""CAWFE-Latte v1.3 separate-support-head ablation tests."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from src.models.cawfe_latte import CAWFELatte, CAWFELatteV11, CAWFELatteV12, CAWFELatteV13  # noqa: E402
from src.models.model_factory import build_model_from_config  # noqa: E402
from src.training.losses import MultiTaskLoss, dilate_fire_mask  # noqa: E402


BATCH = 2
TIME = 5
CHANNELS = 129
HEIGHT = 16
WIDTH = 16


def _x() -> torch.Tensor:
	return torch.randn(BATCH, TIME, CHANNELS, HEIGHT, WIDTH)


def _terrain() -> torch.Tensor:
	return torch.rand(BATCH, 4, HEIGHT, WIDTH)


def _y() -> torch.Tensor:
	y = torch.randn(BATCH, 4, HEIGHT, WIDTH) * 0.1
	y[:, 0:2] = y[:, 0:2].clamp_min(0.0)
	y[:, 2:3] = 0.0
	y[:, 2:3, HEIGHT // 2, WIDTH // 2] = 1.0
	y[:, 3:4] = torch.rand(BATCH, 1, HEIGHT, WIDTH)
	return y


def _config(architecture: str = "cawfe_latte_v1_3") -> dict:
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
			"support_gate": {
				"enabled": True,
				"gate_min": 0.05,
				"gate_max": 1.0,
				"apply_to": ["surface", "canopy", "energy"],
				"support_head": {"in_dim": 64, "out_channels": 1, "kernel_size": 1},
			},
			"auxiliary": {"fire_support_head": {"enabled": True, "source": "support_logits", "weight": 0.2}},
		},
		"cawfe_latte_v1_2": {
			"enabled": True,
			"version": "v1_2_temporal_attention_pooling",
			"temporal_pooling": {"type": "attention", "enabled": True, "dim": 64, "hidden_dim": 16, "initialize_uniform": True},
		},
		"cawfe_latte_v1_3": {
			"enabled": True,
			"version": "v1_3_separate_support_head",
			"support_gate": {
				"enabled": True,
				"source": "separate_support_head",
				"gate_min": 0.05,
				"gate_max": 1.0,
				"apply_to": ["surface", "canopy", "energy"],
				"support_head": {"in_dim": 64, "out_channels": 1, "kernel_size": 1, "bias_init": -2.0},
			},
			"heads": {"regression_activation": "softplus", "regression_bias_init": 0.1, "mask_bias_init": -2.0},
		},
		"energy_release": {"enabled": True, "output_mode": "total", "target_transform": "log1p"},
		"training": {
			"task_type": "multitask",
			"loss": {
				"surface": {"type": "huber", "weight": 1.0, "delta": 1.0},
				"canopy": {"type": "huber", "weight": 1.0, "delta": 1.0},
				"mask": {"type": "bce_dice", "weight": 5.0, "bce_weight": 1.0, "dice_weight": 1.0},
				"energy": {"type": "huber", "weight": 1.0, "delta": 1.0},
				"auxiliary_fire_support": {"enabled": True, "weight": 0.2, "target": "dilated_fire_mask", "dilation_radius": 2, "bce_weight": 1.0, "dice_weight": 1.0},
			},
		},
	}


def test_model_factory_can_create_cawfe_latte_v1_3() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	assert isinstance(model, CAWFELatteV13)


def test_forward_returns_prediction_support_and_aux_with_terrain() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	out = model(_x(), terrain=_terrain())
	assert tuple(out["prediction"].shape) == (BATCH, 4, HEIGHT, WIDTH)
	assert tuple(out["support_logits"].shape) == (BATCH, 1, HEIGHT, WIDTH)
	assert tuple(out["aux_fire_support_logits"].shape) == (BATCH, 1, HEIGHT, WIDTH)


def test_mask_and_support_heads_are_separate_and_mask_channel_is_raw_mask_logits() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	assert model.mask_head is not model.support_head
	features = model(_x(), terrain=_terrain(), return_features=True)
	assert torch.allclose(features["prediction"][:, 2:3], features["raw_mask_logits"])
	assert torch.allclose(features["aux_fire_support_logits"], features["support_logits"])
	assert not torch.allclose(features["prediction"][:, 2:3], features["support_logits"])


def test_support_gate_range_and_temporal_attention_contract() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	features = model(_x(), terrain=_terrain(), return_features=True)
	gate = features["support_gate"]
	assert float(gate.detach().min()) >= 0.05 - 1.0e-6
	assert float(gate.detach().max()) <= 1.0 + 1.0e-6
	alpha = features["temporal_attention_alpha"]
	assert tuple(alpha.shape) == (BATCH, TIME, 1, HEIGHT, WIDTH)
	assert torch.allclose(alpha.sum(dim=1), torch.ones(BATCH, 1, HEIGHT, WIDTH), atol=1.0e-6)


def test_regression_channels_are_softplus_then_gated_and_mask_is_not_gated() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	features = model(_x(), terrain=_terrain(), return_features=True)
	gate = features["support_gate"]
	assert torch.allclose(features["prediction"][:, 0:1], F.softplus(features["raw_surface"]) * gate, atol=1.0e-6)
	assert torch.allclose(features["prediction"][:, 1:2], F.softplus(features["raw_canopy"]) * gate, atol=1.0e-6)
	assert torch.allclose(features["prediction"][:, 3:4], F.softplus(features["raw_energy"]) * gate, atol=1.0e-6)
	assert torch.allclose(features["prediction"][:, 2:3], features["raw_mask_logits"], atol=1.0e-6)


def test_dilated_support_target_expands_fire_mask() -> None:
	mask = torch.zeros(1, 1, 7, 7)
	mask[:, :, 3, 3] = 1.0
	support = dilate_fire_mask(mask, radius=2)
	assert tuple(support.shape) == (1, 1, 7, 7)
	assert torch.all(support >= mask)
	assert float(support.min()) >= 0.0
	assert float(support.max()) <= 1.0
	assert int(support.sum().item()) == 25


def test_loss_uses_dilated_support_target_and_backward_works() -> None:
	config = _config()
	model = build_model_from_config(config, input_channels=CHANNELS)
	criterion = MultiTaskLoss(config)
	out = model(_x(), terrain=_terrain())
	y = _y()
	losses = criterion(out, y)
	assert torch.isfinite(losses["total_loss"])
	assert torch.isfinite(losses["loss_support_total"])
	assert "weighted_loss_support" in losses
	losses["total_loss"].backward()
	grad_sum = sum(float(param.grad.detach().abs().sum().item()) for param in model.parameters() if param.grad is not None)
	assert grad_sum > 0.0

	# Make sure support supervision is not equivalent to the original one-pixel mask target.
	support_target = dilate_fire_mask(y[:, 2:3], radius=2)
	assert float(support_target.sum().item()) > float(y[:, 2:3].sum().item())


def test_v1_v1_1_and_v1_2_are_unaffected() -> None:
	v1 = CAWFELatte(input_channels=CHANNELS, input_sequence_length=TIME, output_channels=4)
	v11 = build_model_from_config(_config("cawfe_latte_v1_1"), input_channels=CHANNELS)
	v12 = build_model_from_config(_config("cawfe_latte_v1_2"), input_channels=CHANNELS)
	assert isinstance(v11, CAWFELatteV11)
	assert isinstance(v12, CAWFELatteV12)
	assert not isinstance(v12, CAWFELatteV13)
	assert tuple(v1(_x(), return_features=True)["prediction"].shape) == (BATCH, 4, HEIGHT, WIDTH)
	assert "temporal_attention_alpha" not in v11(_x(), terrain=_terrain(), return_features=True)
	assert "temporal_attention_alpha" in v12(_x(), terrain=_terrain(), return_features=True)
