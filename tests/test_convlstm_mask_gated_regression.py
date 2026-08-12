from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.convlstm_unet import ConvLSTMUNet
from src.training.losses import MultiTaskLoss


def _loss_config(background_enabled: bool = True) -> dict:
	return {
		"task_type": "multitask",
		"model": {"architecture": "convlstm_unet", "output_channels": 4, "task_type": "multitask"},
		"multitask": {
			"regression_loss": "huber",
			"segmentation_loss": "bce_with_logits",
			"surface_loss_weight": 1.0,
			"canopy_loss_weight": 1.0,
			"segmentation_loss_weight": 1.0,
			"energy_loss_weight": 1.0,
			"active_fire_weight": 2.0,
			"background_weight": 1.0,
			"consumed_fuel_threshold": 0.001,
			"energy_loss": "huber",
			"energy_loss_space": "log",
			"energy_active_weight": 2.0,
			"energy_background_weight": 1.0,
			"energy_active_threshold_MW": 0.001,
		},
		"energy_release": {
			"enabled": True,
			"target_transform": "log1p",
			"predict_total": True,
			"predict_sensible": False,
			"predict_latent": False,
		},
		"training": {
			"loss": {
				"background_suppression": {
					"enabled": background_enabled,
					"weight": 0.05,
					"include_surface": True,
					"include_canopy": True,
					"include_energy": True,
					"include_mask_prob": True,
					"inactive_definition": "combined",
					"consumed_threshold": 0.001,
					"energy_log_threshold": 0.001,
					"mask_threshold": 0.5,
					"reduction": "mean",
				}
			}
		},
	}


def test_convlstm_without_mask_gated_regression_returns_original_shape() -> None:
	model = ConvLSTMUNet(
		input_channels=3,
		output_channels=4,
		convlstm_hidden_dim=4,
		unet_base_channels=4,
		unet_depth=1,
		use_mask_gated_regression=False,
	)
	x = torch.randn(1, 2, 3, 8, 8)

	y = model(x)

	assert tuple(y.shape) == (1, 4, 8, 8)


def test_convlstm_with_mask_gated_regression_returns_multitask_shape() -> None:
	model = ConvLSTMUNet(
		input_channels=3,
		output_channels=4,
		convlstm_hidden_dim=4,
		unet_base_channels=4,
		unet_depth=1,
		use_mask_gated_regression=True,
	)
	x = torch.randn(1, 2, 3, 8, 8)

	y = model(x)

	assert tuple(y.shape) == (1, 4, 8, 8)


def test_mask_channel_remains_logits_when_gating_outputs() -> None:
	model = ConvLSTMUNet(input_channels=1, output_channels=4, use_mask_gated_regression=True)
	raw = torch.zeros(1, 4, 2, 2)
	raw[:, 2] = torch.tensor([[-4.0, 0.0], [4.0, 8.0]])

	y = model._apply_mask_gated_outputs(raw)

	assert torch.equal(y[:, 2], raw[:, 2])


def test_negative_mask_logits_suppress_regression_outputs() -> None:
	model = ConvLSTMUNet(input_channels=1, output_channels=4, use_mask_gated_regression=True, regression_activation="relu")
	raw = torch.zeros(1, 4, 2, 2)
	raw[:, 0] = 10.0
	raw[:, 1] = 10.0
	raw[:, 2] = -30.0
	raw[:, 3] = 10.0

	y = model._apply_mask_gated_outputs(raw)

	assert float(y[:, 0].max().item()) < 1.0e-10
	assert float(y[:, 1].max().item()) < 1.0e-10
	assert float(y[:, 3].max().item()) < 1.0e-10


def test_positive_hard_mask_gate_keeps_activated_regression_outputs() -> None:
	model = ConvLSTMUNet(
		input_channels=1,
		output_channels=4,
		use_mask_gated_regression=True,
		regression_activation="relu",
		mask_gate_mode="hard",
		mask_gate_threshold=0.5,
	)
	raw = torch.zeros(1, 4, 2, 2)
	raw[:, 0] = 2.0
	raw[:, 1] = -3.0
	raw[:, 2] = 30.0
	raw[:, 3] = 4.0

	y = model._apply_mask_gated_outputs(raw)

	assert torch.equal(y[:, 0], torch.full_like(y[:, 0], 2.0))
	assert torch.equal(y[:, 1], torch.zeros_like(y[:, 1]))
	assert torch.equal(y[:, 3], torch.full_like(y[:, 3], 4.0))


def test_background_suppression_loss_is_zero_when_no_inactive_pixels() -> None:
	criterion = MultiTaskLoss(_loss_config(background_enabled=True))
	pred = torch.ones(1, 4, 2, 2)
	true = torch.ones(1, 4, 2, 2)
	true[:, 2] = 1.0

	result = criterion(pred, true)

	assert result["background_suppression_loss"].item() == pytest.approx(0.0)
	assert result["background_suppression_weighted"].item() == pytest.approx(0.0)


def test_background_suppression_loss_is_positive_for_positive_inactive_predictions() -> None:
	criterion = MultiTaskLoss(_loss_config(background_enabled=True))
	pred = torch.ones(1, 4, 2, 2)
	true = torch.zeros(1, 4, 2, 2)

	result = criterion(pred, true)

	assert result["background_suppression_loss"].item() > 0.0
	assert result["background_suppression_weighted"].item() > 0.0


def test_background_suppression_architecture_filter_disables_other_models() -> None:
	config = _loss_config(background_enabled=True)
	config["model"]["architecture"] = "weatherformer_lite"
	config["training"]["loss"]["background_suppression"]["architectures"] = ["convlstm_unet"]
	criterion = MultiTaskLoss(config)
	pred = torch.ones(1, 4, 2, 2)
	true = torch.zeros(1, 4, 2, 2)

	result = criterion(pred, true)

	assert result["background_suppression_loss"].item() == pytest.approx(0.0)
	assert result["background_suppression_weighted"].item() == pytest.approx(0.0)


def test_output_bias_initialization_sets_final_conv_bias_channels() -> None:
	model = ConvLSTMUNet(
		input_channels=1,
		output_channels=4,
		output_bias_init={
			"enabled": True,
			"regression_bias": -2.0,
			"mask_bias": -3.0,
			"energy_bias": -4.0,
		},
	)
	bias = model.spatial_decoder.outc.proj.bias.detach()

	assert model.output_bias_init_applied is True
	assert bias[0].item() == pytest.approx(-2.0)
	assert bias[1].item() == pytest.approx(-2.0)
	assert bias[2].item() == pytest.approx(-3.0)
	assert bias[3].item() == pytest.approx(-4.0)
