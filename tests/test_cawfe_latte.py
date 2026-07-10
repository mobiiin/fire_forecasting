from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.cawfe_latte import CAWFELatte
from src.models.cawfe_latte_constraints import PhysicalOutputConstraintLayer
from src.models.cawfe_latte_operator import NeuralOperatorBottleneck
from src.models.cawfe_latte_wind import WindGuidedDirectionalModule
from src.models.input_adapters import adapt_input_for_architecture
from src.models.model_factory import build_model_from_config
from src.training.losses import get_loss_function


def _config() -> dict:
	return {
		"task_type": "multitask",
		"input_sequence_length": 6,
		"model": {
			"architecture": "cawfe_latte",
			"name": "cawfe_latte",
			"input_channels": 129,
			"output_channels": 4,
		},
		"cawfe_latte": {
			"input_sequence_length": 6,
			"patch_size": 64,
			"embed_dim": 32,
			"atm_embed_dim": 16,
			"fire_embed_dim": 16,
			"fused_dim": 32,
			"backbone_dim": 32,
			"bottleneck_dim": 64,
			"atmosphere_start_channel": 0,
			"atmosphere_num_levels": 8,
			"atmosphere_vars_per_level": 10,
			"atmosphere_num_channels": 80,
			"flux_channels": [80, 81, 82, 83],
			"fuel_channels": [84, 85],
			"engineered_start_channel": 86,
			"engineered_end_channel": 128,
			"wind_direction_cos_channels": [],
			"wind_direction_sin_channels": [],
			"wind_speed_channels": [],
			"use_vertical_atmosphere_encoder": True,
			"vertical_encoder_type": "attention",
			"use_fire_fuel_encoder": True,
			"use_fire_front_gate": True,
			"use_wind_guided_directional_module": True,
			"wind_guidance_mode": "feature_modulation",
			"wind_guidance_strength": 1.0,
			"wind_guidance_hidden_dim": 16,
			"use_neural_operator_bottleneck": True,
			"neural_operator_type": "afno",
			"neural_operator_depth": 1,
			"neural_operator_num_blocks": 4,
			"neural_operator_sparsity_threshold": 0.01,
			"neural_operator_hard_thresholding_fraction": 1.0,
			"neural_operator_hidden_factor": 1,
			"neural_operator_force_float32_fft": True,
			"backbone_type": "hybrid_transformer_mamba",
			"backbone_depths": [1, 1],
			"num_heads": [4, 4],
			"window_size": 8,
			"shifted_window": True,
			"mamba_backend": "fallback",
			"mamba_d_state": 8,
			"mamba_d_conv": 3,
			"mamba_expand": 1,
			"mamba_scan_mode": "tri_axis",
			"fire_gate_hidden_dim": 16,
			"fire_gate_strength": 1.0,
			"fire_gate_mode": "multiplicative",
			"fire_gate_channels": {"flux": True, "fuel": True, "engineered": True},
			"temporal_readout": "attention_pool",
			"decoder_channels": [64, 32, 16],
			"decoder_task_heads": "separate",
			"use_skip_connections": True,
			"use_bottleneck_skip": True,
			"upsample_mode": "bilinear",
			"use_physical_output_constraints": True,
			"constrain_consumed_nonnegative": True,
			"constrain_energy_nonnegative": True,
			"mask_output_is_logits": True,
			"dropout": 0.0,
			"attention_dropout": 0.0,
			"drop_path": 0.0,
			"mlp_ratio": 2.0,
			"gradient_checkpointing": False,
			"required_patch_divisibility": 16,
			"return_aux_default": False,
			"save_module_maps": False,
		},
		"multitask": {
			"regression_loss": "huber",
			"segmentation_loss": "bce_with_logits",
			"surface_loss_weight": 1.0,
			"canopy_loss_weight": 1.0,
			"segmentation_loss_weight": 1.0,
			"energy_loss_weight": 1.0,
			"active_fire_weight": 2.0,
			"background_weight": 1.0,
			"consumed_fuel_threshold": 0.01,
			"energy_loss": "huber",
			"energy_loss_space": "log",
			"energy_active_weight": 2.0,
			"energy_background_weight": 1.0,
			"energy_active_threshold_MW": 0.001,
		},
		"energy_release": {
			"enabled": True,
			"target_transform": "log1p",
			"inverse_transform": "expm1",
			"predict_total": True,
			"predict_sensible": False,
			"predict_latent": False,
		},
	}


def test_cawfe_latte_forward_shape() -> None:
	model = build_model_from_config(_config(), input_channels=129)
	x = torch.randn(2, 6, 129, 64, 64)
	y = model(x)
	assert tuple(y.shape) == (2, 4, 64, 64)


def test_wind_guided_directional_module_preserves_shape_and_has_finite_wind() -> None:
	module = WindGuidedDirectionalModule(
		input_channels=129,
		feature_channels=32,
		hidden_dim=16,
		guidance_strength=1.0,
		mode="feature_modulation",
	)
	fused = torch.randn(2, 6, 32, 16, 16)
	raw = torch.randn(2, 6, 129, 16, 16)
	guided, aux = module(fused, raw)
	assert tuple(guided.shape) == tuple(fused.shape)
	for key in ("wind_speed", "wind_cos", "wind_sin"):
		assert torch.all(torch.isfinite(aux[key]))


def test_neural_operator_bottleneck_preserves_shape() -> None:
	operator = NeuralOperatorBottleneck(
		channels=64,
		operator_type="afno",
		depth=1,
		num_blocks=4,
		force_float32_fft=True,
	)
	x = torch.randn(2, 6, 64, 32, 32)
	y = operator(x)
	assert tuple(y.shape) == tuple(x.shape)


def test_neural_operator_bottleneck_accepts_low_precision_input() -> None:
	operator = NeuralOperatorBottleneck(
		channels=16,
		operator_type="afno",
		depth=1,
		num_blocks=4,
		force_float32_fft=True,
	)
	x = torch.randn(1, 2, 16, 8, 8).half()
	y = operator(x)
	assert tuple(y.shape) == tuple(x.shape)
	assert y.dtype == x.dtype


def test_cawfe_latte_forward_return_aux() -> None:
	model = build_model_from_config(_config(), input_channels=129)
	x = torch.randn(2, 6, 129, 64, 64)
	pred, aux = model(x, return_aux=True)
	assert tuple(pred.shape) == (2, 4, 64, 64)
	assert aux["fire_gate_map"] is not None
	assert aux["wind_guidance_map"] is not None
	assert "wind_direction_summary" in aux
	assert "neural_operator_energy" in aux


def test_model_factory_returns_cawfe_latte() -> None:
	model = build_model_from_config(_config(), input_channels=129)
	assert isinstance(model, CAWFELatte)


def test_input_adapter_returns_unchanged_sequence() -> None:
	x = torch.randn(2, 6, 129, 64, 64)
	y = adapt_input_for_architecture(x, "cawfe_latte")
	assert y is x


def test_physical_constraints_leave_mask_unconstrained() -> None:
	layer = PhysicalOutputConstraintLayer(
		constrain_consumed_nonnegative=True,
		constrain_energy_nonnegative=True,
		mask_output_is_logits=True,
	)
	pred = torch.randn(2, 4, 8, 8) - 2.0
	constrained = layer(pred)
	assert torch.all(constrained[:, 0] >= 0)
	assert torch.all(constrained[:, 1] >= 0)
	assert torch.all(constrained[:, 3] >= 0)
	assert torch.equal(constrained[:, 2], pred[:, 2])


def test_one_training_step_has_finite_gradients() -> None:
	config = _config()
	model = build_model_from_config(config, input_channels=129)
	criterion = get_loss_function(config)
	x = torch.randn(2, 6, 129, 64, 64)
	target = torch.randn(2, 4, 64, 64)
	target[:, 2] = torch.randint(0, 2, size=(2, 64, 64)).to(torch.float32)
	prediction = model(x)
	loss_result = criterion(prediction, target)
	loss = loss_result["total_loss"] if isinstance(loss_result, dict) else loss_result
	loss.backward()
	assert torch.isfinite(loss)
	for parameter in model.parameters():
		if parameter.grad is not None:
			assert torch.all(torch.isfinite(parameter.grad))
