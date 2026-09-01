"""Active CAWFE-Latte architecture cleanup tests."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.config import load_config  # noqa: E402
from src.models.architecture_registry import get_architecture_spec  # noqa: E402
from src.models.cawfe_latte import CAWFELatte  # noqa: E402
from src.models.model_factory import build_model_from_config  # noqa: E402
from src.training.losses import MultiTaskLoss  # noqa: E402


BATCH = 2
TIME = 5
CHANNELS = 129
HEIGHT = 16
WIDTH = 16
ARCHIVED = ("cawfe_latte_v1_1", "cawfe_latte_v1_2", "cawfe_latte_v1_3")


def _config(architecture: str = "cawfe_latte") -> dict:
	return {
		"task_type": "multitask",
		"model": {"architecture": architecture, "input_channels": CHANNELS, "output_channels": 4},
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


def _x() -> torch.Tensor:
	return torch.randn(BATCH, TIME, CHANNELS, HEIGHT, WIDTH)


def _y() -> torch.Tensor:
	y = torch.rand(BATCH, 4, HEIGHT, WIDTH)
	y[:, 2:3] = (torch.rand(BATCH, 1, HEIGHT, WIDTH) > 0.7).float()
	return y


def test_cawfe_latte_is_active_and_forward_works() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	assert isinstance(model, CAWFELatte)
	out = model(_x())
	assert set(out) == {"prediction", "aux_fire_support_logits"}
	assert tuple(out["prediction"].shape) == (BATCH, 4, HEIGHT, WIDTH)
	assert tuple(out["aux_fire_support_logits"].shape) == (BATCH, 1, HEIGHT, WIDTH)


@pytest.mark.parametrize("architecture", ARCHIVED)
def test_archived_cawfe_latte_variants_are_not_active(architecture: str) -> None:
	with pytest.raises(ValueError, match="archived"):
		build_model_from_config(_config(architecture), input_channels=CHANNELS)
	with pytest.raises(KeyError, match="archived"):
		get_architecture_spec(architecture)


def test_cawfe_latte_v1_config_still_loads_and_selects_original_architecture() -> None:
	config = load_config("configs/experiments/cawfe_latte_v1.yaml")
	assert config["model"]["architecture"] == "cawfe_latte"
	model = build_model_from_config(config, input_channels=CHANNELS)
	assert isinstance(model, CAWFELatte)


def test_original_cawfe_latte_training_loss_still_works() -> None:
	model = build_model_from_config(_config(), input_channels=CHANNELS)
	criterion = MultiTaskLoss(_config())
	losses = criterion(model(_x()), _y())
	assert torch.isfinite(losses["total_loss"])
	assert "loss_aux_fire_support_total" in losses
	losses["total_loss"].backward()
	grad_sum = sum(float(param.grad.detach().abs().sum().item()) for param in model.parameters() if param.grad is not None)
	assert grad_sum > 0.0
