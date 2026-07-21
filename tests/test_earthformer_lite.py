from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.earthformer_blocks import AxialCuboidAttentionBlock
from src.models.earthformer_lite import EarthformerLite
from src.models.model_factory import build_model_from_config
from src.training.losses import get_loss_function


def _config() -> dict:
	return {
		"task_type": "multitask",
		"input_sequence_length": 5,
		"model": {
			"architecture": "earthformer_lite",
			"name": "earthformer_lite",
			"input_channels": 129,
			"output_channels": 4,
		},
		"earthformer_lite": {
			"input_sequence_length": 5,
			"patch_size": 64,
			"embed_dim": 32,
			"depths": [1, 1],
			"num_heads": [4, 4],
			"mlp_ratio": 2.0,
			"dropout": 0.0,
			"attention_dropout": 0.0,
			"drop_path": 0.0,
			"use_global_vectors": True,
			"num_global_vectors": 4,
			"use_time_pos_embed": True,
			"use_space_pos_embed": True,
			"gradient_checkpointing": False,
			"temporal_readout": "attention_pool",
			"downsample_stages": 2,
			"patch_merge_factor": 2,
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


def test_earthformer_lite_forward_shape() -> None:
	config = _config()
	model = build_model_from_config(config, input_channels=129)
	x = torch.randn(2, 5, 129, 64, 64)
	y = model(x)
	assert tuple(y.shape) == (2, 4, 64, 64)


def test_axial_cuboid_attention_block_preserves_shape() -> None:
	block = AxialCuboidAttentionBlock(
		dim=64,
		num_heads=4,
		mlp_ratio=2.0,
		dropout=0.0,
		attention_dropout=0.0,
		drop_path=0.0,
		use_global_vectors=True,
		num_global_vectors=4,
	)
	x = torch.randn(2, 5, 64, 64, 64)
	y = block(x)
	assert tuple(y.shape) == tuple(x.shape)


def test_patch_size_divisibility_validation() -> None:
	with pytest.raises(ValueError):
		EarthformerLite(
			input_channels=129,
			output_channels=4,
			input_sequence_length=5,
			patch_size=72,
			embed_dim=32,
			depths=[1, 1],
			num_heads=[4, 4],
			mlp_ratio=2.0,
			dropout=0.0,
			attention_dropout=0.0,
			drop_path=0.0,
			use_global_vectors=False,
			num_global_vectors=0,
			use_time_pos_embed=True,
			use_space_pos_embed=True,
			gradient_checkpointing=False,
			required_patch_divisibility=16,
		)


def test_model_factory_returns_earthformer_lite() -> None:
	model = build_model_from_config(_config(), input_channels=129)
	assert isinstance(model, EarthformerLite)


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
