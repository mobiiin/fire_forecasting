from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.cawfe_latte_backbone import LatteHybridBlock
from src.models.cawfe_latte_constraints import PhysicalOutputConstraintLayer
from src.models.cawfe_latte_fire import FireFrontAttentionGate, FireFuelStateEncoder
from src.models.cawfe_latte_lite import CAWFELatteLite
from src.models.cawfe_latte_vertical import VerticalAtmosphereEncoder
from src.models.input_adapters import adapt_input_for_architecture
from src.models.model_factory import build_model_from_config
from src.training.losses import get_loss_function


def _config() -> dict:
	return {
		"task_type": "multitask",
		"input_sequence_length": 5,
		"model": {
			"architecture": "cawfe_latte_lite",
			"name": "cawfe_latte_lite",
			"input_channels": 129,
			"output_channels": 4,
		},
		"cawfe_latte_lite": {
			"input_sequence_length": 5,
			"patch_size": 64,
			"embed_dim": 32,
			"atm_embed_dim": 16,
			"fire_embed_dim": 16,
			"fused_dim": 32,
			"backbone_dim": 32,
			"atmosphere_start_channel": 0,
			"atmosphere_num_levels": 8,
			"atmosphere_vars_per_level": 10,
			"atmosphere_num_channels": 80,
			"flux_channels": [80, 81, 82, 83],
			"fuel_channels": [84, 85],
			"engineered_start_channel": 86,
			"engineered_end_channel": 128,
			"use_vertical_atmosphere_encoder": True,
			"vertical_encoder_type": "attention",
			"use_fire_fuel_encoder": True,
			"use_fire_front_gate": True,
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
			"decoder_channels": [64, 32],
			"decoder_task_heads": "separate",
			"use_skip_connections": True,
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


def test_cawfe_latte_lite_forward_shape() -> None:
	model = build_model_from_config(_config(), input_channels=129)
	x = torch.randn(2, 5, 129, 64, 64)
	y = model(x)
	assert tuple(y.shape) == (2, 4, 64, 64)


def test_vertical_atmosphere_encoder_shape() -> None:
	encoder = VerticalAtmosphereEncoder(
		num_levels=8,
		vars_per_level=10,
		atm_embed_dim=16,
		encoder_type="attention",
		num_heads=4,
	)
	x = torch.randn(2, 5, 80, 16, 16)
	y = encoder(x)
	assert tuple(y.shape) == (2, 5, 16, 16, 16)


def test_vertical_atmosphere_attention_chunking_matches_unchunked() -> None:
	torch.manual_seed(7)
	full_encoder = VerticalAtmosphereEncoder(
		num_levels=8,
		vars_per_level=10,
		atm_embed_dim=16,
		encoder_type="attention",
		num_heads=4,
		attention_chunk_size=0,
	)
	chunked_encoder = VerticalAtmosphereEncoder(
		num_levels=8,
		vars_per_level=10,
		atm_embed_dim=16,
		encoder_type="attention",
		num_heads=4,
		attention_chunk_size=7,
	)
	chunked_encoder.load_state_dict(full_encoder.state_dict())
	full_encoder.eval()
	chunked_encoder.eval()
	x = torch.randn(2, 3, 80, 5, 4)

	with torch.no_grad():
		full_y = full_encoder(x)
		chunked_y = chunked_encoder(x)

	assert torch.allclose(chunked_y, full_y, atol=1e-6, rtol=1e-5)


def test_fire_fuel_state_encoder_shape() -> None:
	encoder = FireFuelStateEncoder(
		input_channels=129,
		fire_embed_dim=16,
		flux_channels=[80, 81, 82, 83],
		fuel_channels=[84, 85],
		engineered_start_channel=86,
		engineered_end_channel=128,
	)
	x = torch.randn(2, 5, 129, 16, 16)
	y = encoder(x)
	assert tuple(y.shape) == (2, 5, 16, 16, 16)


def test_fire_front_attention_gate_shapes() -> None:
	gate = FireFrontAttentionGate(
		gate_input_channels=49,
		fused_channels=32,
		hidden_dim=16,
		gate_strength=1.0,
		gate_mode="multiplicative",
	)
	fused = torch.randn(2, 5, 32, 16, 16)
	gate_input = torch.randn(2, 5, 49, 16, 16)
	gated, a_fire = gate(fused, gate_input)
	assert tuple(gated.shape) == tuple(fused.shape)
	assert tuple(a_fire.shape) == (2, 5, 1, 16, 16)


def test_latte_hybrid_block_preserves_shape() -> None:
	block = LatteHybridBlock(
		channels=32,
		num_heads=4,
		window_size=8,
		shifted_window=True,
		backbone_type="hybrid_transformer_mamba",
		mamba_backend="fallback",
		mamba_d_state=8,
		mamba_d_conv=3,
		mamba_expand=1,
		mamba_scan_mode="tri_axis",
		mlp_ratio=2.0,
		dropout=0.0,
		attention_dropout=0.0,
		drop_path=0.0,
	)
	x = torch.randn(2, 5, 32, 16, 16)
	y = block(x)
	assert tuple(y.shape) == tuple(x.shape)


def test_model_factory_returns_cawfe_latte_lite() -> None:
	model = build_model_from_config(_config(), input_channels=129)
	assert isinstance(model, CAWFELatteLite)


def test_input_adapter_returns_unchanged_sequence() -> None:
	x = torch.randn(2, 5, 129, 64, 64)
	y = adapt_input_for_architecture(x, "cawfe_latte_lite")
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


def test_forward_return_aux() -> None:
	model = build_model_from_config(_config(), input_channels=129)
	x = torch.randn(2, 5, 129, 64, 64)
	pred, aux = model(x, return_aux=True)
	assert tuple(pred.shape) == (2, 4, 64, 64)
	assert aux["fire_gate_map"] is not None
	assert "temporal_attention_weights" in aux
	assert "module_enabled_flags" in aux


def test_one_training_step_has_finite_gradients() -> None:
	config = _config()
	model = build_model_from_config(config, input_channels=129)
	criterion = get_loss_function(config)
	x = torch.randn(2, 5, 129, 64, 64)
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
