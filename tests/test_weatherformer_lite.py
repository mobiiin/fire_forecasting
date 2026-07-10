from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.input_adapters import adapt_input_for_architecture
from src.models.model_factory import build_model_from_config
from src.models.weatherformer_blocks import FactorizedWeatherFormerBlock
from src.models.weatherformer_lite import WeatherFormerLite
from src.models.window_attention import window_partition, window_reverse
from src.training.losses import get_loss_function


def _config() -> dict:
	return {
		"task_type": "multitask",
		"input_sequence_length": 6,
		"model": {
			"architecture": "weatherformer_lite",
			"name": "weatherformer_lite",
			"input_channels": 129,
			"output_channels": 4,
		},
		"weatherformer_lite": {
			"input_sequence_length": 6,
			"patch_size": 64,
			"embed_dim": 32,
			"encoder_channels": [32, 64],
			"decoder_channels": [64, 32],
			"depths": [1, 1],
			"num_heads": [4, 4],
			"mlp_ratio": 2.0,
			"use_channel_scaler": True,
			"use_feature_gate": True,
			"scaler_init": 1.0,
			"use_time_pos_embed": True,
			"use_2d_space_pos_embed": True,
			"use_fourier_space_encoding": True,
			"attention_type": "factorized",
			"temporal_attention": True,
			"spatial_attention": "window",
			"window_size": 8,
			"shifted_window": True,
			"use_global_tokens": True,
			"num_global_tokens": 2,
			"downsample_stages": 2,
			"patch_merge_factor": 2,
			"temporal_readout": "attention_pool",
			"use_unet_decoder": True,
			"use_skip_connections": True,
			"upsample_mode": "bilinear",
			"dropout": 0.0,
			"attention_dropout": 0.0,
			"drop_path": 0.0,
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


def test_weatherformer_lite_forward_shape() -> None:
	model = build_model_from_config(_config(), input_channels=129)
	x = torch.randn(2, 6, 129, 64, 64)
	y = model(x)
	assert tuple(y.shape) == (2, 4, 64, 64)


def test_factorized_weatherformer_block_preserves_shape() -> None:
	block = FactorizedWeatherFormerBlock(
		channels=64,
		num_heads=4,
		mlp_ratio=2.0,
		window_size=8,
		shifted_window=True,
		use_global_tokens=True,
		num_global_tokens=2,
		dropout=0.0,
		attention_dropout=0.0,
		drop_path=0.0,
		gradient_checkpointing=False,
	)
	x = torch.randn(2, 6, 64, 16, 16)
	y = block(x)
	assert tuple(y.shape) == tuple(x.shape)


def test_window_partition_roundtrip() -> None:
	x = torch.randn(2, 16, 16, 8)
	windows = window_partition(x, window_size=8)
	restored = window_reverse(windows, window_size=8, height=16, width=16)
	assert torch.equal(restored, x)


def test_patch_size_divisibility_validation() -> None:
	with pytest.raises(ValueError):
		WeatherFormerLite(
			input_channels=129,
			output_channels=4,
			input_sequence_length=6,
			patch_size=72,
			embed_dim=32,
			encoder_channels=[32, 64],
			decoder_channels=[64, 32],
			depths=[1, 1],
			num_heads=[4, 4],
			mlp_ratio=2.0,
			use_channel_scaler=True,
			use_feature_gate=True,
			scaler_init=1.0,
			use_time_pos_embed=True,
			use_2d_space_pos_embed=True,
			use_fourier_space_encoding=True,
			window_size=8,
			shifted_window=True,
			use_global_tokens=True,
			num_global_tokens=2,
			temporal_readout="attention_pool",
			dropout=0.0,
			attention_dropout=0.0,
			drop_path=0.0,
			gradient_checkpointing=False,
			required_patch_divisibility=16,
		)


def test_model_factory_returns_weatherformer_lite() -> None:
	model = build_model_from_config(_config(), input_channels=129)
	assert isinstance(model, WeatherFormerLite)


def test_input_adapter_returns_unchanged_sequence() -> None:
	x = torch.randn(2, 6, 129, 64, 64)
	y = adapt_input_for_architecture(x, "weatherformer_lite")
	assert y is x


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
