"""End-to-end CAWFE-Latte v1 tests."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.models.cawfe_latte import CAWFELatte  # noqa: E402
from src.models.model_factory import build_model_from_config  # noqa: E402
from src.training.losses import MultiTaskLoss  # noqa: E402


BATCH = 2
TIME = 5
CHANNELS = 129
HEIGHT = 16
WIDTH = 16


def _x() -> torch.Tensor:
	return torch.randn(BATCH, TIME, CHANNELS, HEIGHT, WIDTH)


def _y() -> torch.Tensor:
	y = torch.randn(BATCH, 4, HEIGHT, WIDTH) * 0.1
	y[:, 2:3] = (torch.rand(BATCH, 1, HEIGHT, WIDTH) > 0.7).float()
	y[:, 3:4] = torch.rand(BATCH, 1, HEIGHT, WIDTH)
	return y


def _config() -> dict:
	return {
		"task_type": "multitask",
		"model": {"architecture": "cawfe_latte", "input_channels": CHANNELS, "output_channels": 4},
		"input_sequence_length": TIME,
		"cawfe_latte": {"input_channels": CHANNELS, "input_sequence_length": TIME, "output_channels": 4, "output_dim": 64},
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


def test_cawfe_latte_v1_forward_returns_prediction_and_aux() -> None:
	model = CAWFELatte(input_channels=CHANNELS, input_sequence_length=TIME, output_channels=4)
	out = model(_x())
	assert set(out) == {"prediction", "aux_fire_support_logits"}
	assert tuple(out["prediction"].shape) == (BATCH, 4, HEIGHT, WIDTH)
	assert tuple(out["aux_fire_support_logits"].shape) == (BATCH, 1, HEIGHT, WIDTH)


def test_cawfe_latte_v1_return_features() -> None:
	model = CAWFELatte(input_channels=CHANNELS, input_sequence_length=TIME, output_channels=4)
	features = model(_x(), return_features=True)
	for key in ("atmosphere", "wind", "fire_fuel", "flux_energy", "fused", "fused_grid", "local"):
		assert tuple(features[key].shape) == (BATCH, TIME, 64, HEIGHT, WIDTH)
	for key in ("aligned_atmosphere", "aligned_wind", "aligned_fire_fuel", "aligned_flux_energy", "fused_tokens"):
		assert tuple(features[key].shape) == (BATCH, TIME, HEIGHT * WIDTH, 64)
	assert tuple(features["aggregated"].shape) == (BATCH, 64, HEIGHT, WIDTH)
	assert tuple(features["decoded"].shape) == (BATCH, 64, HEIGHT, WIDTH)
	assert tuple(features["prediction"].shape) == (BATCH, 4, HEIGHT, WIDTH)


def test_mask_prediction_channel_is_logits() -> None:
	model = CAWFELatte(input_channels=CHANNELS, input_sequence_length=TIME, output_channels=4)
	out = model(_x())["prediction"]
	mask_logits = out[:, 2]
	assert bool(((mask_logits < 0.0) | (mask_logits > 1.0)).any().item())


def test_loss_accepts_cawfe_latte_dict_output() -> None:
	model = CAWFELatte(input_channels=CHANNELS, input_sequence_length=TIME, output_channels=4)
	criterion = MultiTaskLoss(_config())
	losses = criterion(model(_x()), _y())
	assert torch.isfinite(losses["total_loss"])
	assert "loss_aux_fire_support_total" in losses


def test_loss_accepts_plain_tensor_output() -> None:
	config = _config()
	config["model"]["architecture"] = "convlstm_unet"
	criterion = MultiTaskLoss(config)
	losses = criterion(torch.randn(BATCH, 4, HEIGHT, WIDTH), _y())
	assert torch.isfinite(losses["total_loss"])
	assert "loss_aux_fire_support_total" not in losses


def test_backward_pass_reaches_cawfe_latte_parameters() -> None:
	model = CAWFELatte(input_channels=CHANNELS, input_sequence_length=TIME, output_channels=4)
	criterion = MultiTaskLoss(_config())
	loss = criterion(model(_x()), _y())["total_loss"]
	loss.backward()
	grad_sum = sum(float(param.grad.detach().abs().sum().item()) for param in model.parameters() if param.grad is not None)
	assert grad_sum > 0.0


def test_model_factory_can_create_cawfe_latte_v1() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	assert isinstance(model, CAWFELatte)
	out = model(_x())["prediction"]
	assert tuple(out.shape[-2:]) == (HEIGHT, WIDTH)
