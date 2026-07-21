from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.input_adapters import adapt_input_for_architecture
from src.models.mamba_backend import build_mamba_layer
from src.models.model_factory import build_model_from_config
from src.models.scan_routes import flatten_by_route, unflatten_by_route
from src.models.st_mamba_lite import STMamba
from src.models.st_mamba_lite_blocks import STMambaBlock, SpatialTemporalRouteMamba
from src.training.losses import get_loss_function


def _config() -> dict:
	return {
		"task_type": "multitask",
		"input_sequence_length": 5,
		"model": {
			"architecture": "st_mamba_lite",
			"name": "st_mamba_lite",
			"input_channels": 129,
			"output_channels": 4,
		},
		"st_mamba_lite": {
			"input_sequence_length": 5,
			"patch_size": 64,
			"embed_dim": 32,
			"encoder_channels": [32, 64],
			"decoder_channels": [64, 32],
			"depths": [1, 1],
			"mamba_backend": "fallback",
			"d_state": 16,
			"d_conv": 4,
			"expand": 2,
			"scan_mode": "route_pair",
			"scan_routes": ["HVT", "TVH"],
			"bidirectional_scan": True,
			"use_depthwise_conv3d": True,
			"depthwise_conv3d_kernel_size": [3, 3, 3],
			"use_st_mixer": True,
			"st_mixer_sequence_order": "THW",
			"temporal_readout": "attention_pool",
			"use_unet_decoder": True,
			"use_skip_connections": True,
			"upsample_mode": "bilinear",
			"dropout": 0.0,
			"drop_path": 0.0,
			"mlp_ratio": 2.0,
			"use_adaln_conditioning": False,
			"use_fire_static_embedding": False,
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


def test_st_mamba_lite_forward_shape() -> None:
	model = build_model_from_config(_config(), input_channels=129)
	x = torch.randn(2, 5, 129, 64, 64)
	y = model(x)
	assert tuple(y.shape) == (2, 4, 64, 64)


def test_st_mamba_block_preserves_shape() -> None:
	block = STMambaBlock(
		channels=64,
		mamba_backend="fallback",
		d_state=16,
		d_conv=4,
		expand=2,
		scan_mode="route_pair",
		scan_routes=["HVT", "TVH"],
		bidirectional_scan=True,
		use_depthwise_conv3d=True,
		depthwise_conv3d_kernel_size=(3, 3, 3),
		use_st_mixer=True,
		st_mixer_sequence_order="THW",
		dropout=0.0,
		drop_path=0.0,
		mlp_ratio=2.0,
		gradient_checkpointing=False,
	)
	x = torch.randn(2, 5, 64, 64, 64)
	y = block(x)
	assert tuple(y.shape) == tuple(x.shape)


@pytest.mark.parametrize("route", ["HVT", "TVH"])
def test_scan_route_roundtrip(route: str) -> None:
	x = torch.randn(2, 5, 8, 4, 5)
	sequence = flatten_by_route(x, route)
	restored = unflatten_by_route(sequence, route, tuple(int(value) for value in x.shape))
	assert torch.equal(restored, x)


def test_bidirectional_scan_preserves_shape() -> None:
	layer = SpatialTemporalRouteMamba(
		channels=32,
		mamba_backend="fallback",
		d_state=16,
		d_conv=4,
		expand=2,
		scan_mode="route_pair",
		scan_routes=["HVT", "TVH"],
		bidirectional_scan=True,
		use_st_mixer=False,
	)
	x = torch.randn(2, 5, 32, 8, 8)
	y = layer(x)
	assert tuple(y.shape) == tuple(x.shape)


def test_model_factory_returns_st_mamba_lite() -> None:
	model = build_model_from_config(_config(), input_channels=129)
	assert isinstance(model, STMamba)


def test_model_factory_accepts_cawfe_st_mamba_alias() -> None:
	config = _config()
	config["model"]["architecture"] = "cawfe_st_mamba"
	config["model"]["name"] = "cawfe_st_mamba"
	model = build_model_from_config(config, input_channels=129)
	assert isinstance(model, STMamba)


def test_input_adapter_returns_unchanged_sequence() -> None:
	x = torch.randn(2, 5, 129, 64, 64)
	y = adapt_input_for_architecture(x, "st_mamba_lite")
	assert y is x


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


def test_missing_mamba_ssm_backend_raises_import_error() -> None:
	with pytest.raises(ImportError, match="mamba-ssm is not installed"):
		build_mamba_layer(
			d_model=32,
			d_state=16,
			d_conv=4,
			expand=2,
			backend="mamba_ssm",
			dropout=0.0,
		)
