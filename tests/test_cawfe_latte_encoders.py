"""Shape tests for the fresh CAWFE-Latte encoders and fusion stem."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.models.cawfe_latte import (  # noqa: E402
	AtmosphereEncoder,
	CAWFELatte,
	FireFuelEncoder,
	FireQueryCrossAttentionFusion,
	FluxEnergyEncoder,
	MultimodalAlignment,
	WindEncoder,
)
from src.models.model_factory import build_model_from_config  # noqa: E402


BATCH = 2
TIME = 5
CHANNELS = 129
HEIGHT = 16
WIDTH = 16
DIM = 64


def _input() -> torch.Tensor:
	return torch.randn(BATCH, TIME, CHANNELS, HEIGHT, WIDTH)


def test_atmosphere_encoder_output_shape() -> None:
	encoder = AtmosphereEncoder(out_dim=DIM, hidden_dim=DIM, num_blocks=1)
	assert tuple(encoder(_input()).shape) == (BATCH, TIME, DIM, HEIGHT, WIDTH)


def test_wind_encoder_output_shape() -> None:
	encoder = WindEncoder(input_channels=CHANNELS, out_dim=DIM, hidden_dim=DIM, num_blocks=1, channel_names={})
	assert tuple(encoder(_input()).shape) == (BATCH, TIME, DIM, HEIGHT, WIDTH)


def test_fire_fuel_encoder_output_shape() -> None:
	encoder = FireFuelEncoder(input_channels=CHANNELS, out_dim=DIM, hidden_dim=DIM, num_blocks=2, channel_names={})
	assert tuple(encoder(_input()).shape) == (BATCH, TIME, DIM, HEIGHT, WIDTH)


def test_flux_energy_encoder_output_shape() -> None:
	encoder = FluxEnergyEncoder(input_channels=CHANNELS, out_dim=DIM, hidden_dim=DIM, num_blocks=1, channel_names={})
	assert tuple(encoder(_input()).shape) == (BATCH, TIME, DIM, HEIGHT, WIDTH)


def test_fusion_output_shape_and_attention_shape() -> None:
	fusion = FireQueryCrossAttentionFusion(dim=DIM, num_heads=4, dropout=0.0)
	tokens = [torch.randn(BATCH, TIME, HEIGHT * WIDTH, DIM) for _ in range(4)]
	z, weights = fusion(*tokens, return_attention=True)
	assert tuple(z.shape) == (BATCH, TIME, HEIGHT * WIDTH, DIM)
	assert tuple(weights.shape) == (BATCH, TIME, HEIGHT * WIDTH, 3)


def test_cawfe_latte_forward_return_features_keys() -> None:
	model = CAWFELatte(input_channels=CHANNELS, input_sequence_length=TIME, output_dim=DIM)
	features = model(_input(), return_features=True)
	for key in ("atmosphere", "wind", "fire_fuel", "flux_energy", "fused", "fused_grid", "local"):
		assert tuple(features[key].shape) == (BATCH, TIME, DIM, HEIGHT, WIDTH)
	for key in ("aligned_atmosphere", "aligned_wind", "aligned_fire_fuel", "aligned_flux_energy", "fused_tokens"):
		assert tuple(features[key].shape) == (BATCH, TIME, HEIGHT * WIDTH, DIM)
	assert features["spatial_shape"] == (HEIGHT, WIDTH)


def test_cawfe_latte_attention_weights_are_driver_modalities() -> None:
	model = CAWFELatte(input_channels=CHANNELS, input_sequence_length=TIME, output_dim=DIM)
	features = model(_input(), return_features=True, return_attention=True)
	assert tuple(features["fusion_attention"].shape) == (BATCH, TIME, HEIGHT * WIDTH, 3)
	assert FireQueryCrossAttentionFusion.modalities == ("atmosphere", "wind", "flux_energy")


def test_cawfe_latte_too_few_channels_raises_clear_error() -> None:
	model = CAWFELatte(input_channels=CHANNELS, input_sequence_length=TIME, output_dim=DIM)
	with pytest.raises(ValueError, match="requires at least 86 input channels"):
		model(torch.randn(BATCH, TIME, 85, HEIGHT, WIDTH))


def test_model_factory_builds_new_cawfe_latte_stem() -> None:
	model = build_model_from_config(
		{
			"model": {"architecture": "cawfe_latte", "input_channels": CHANNELS, "output_channels": 4},
			"input_sequence_length": TIME,
			"cawfe_latte": {"input_channels": CHANNELS, "input_sequence_length": TIME, "output_dim": DIM},
		},
		input_channels=CHANNELS,
	)
	assert isinstance(model, CAWFELatte)
	output = model(_input())
	assert tuple(output["prediction"].shape) == (BATCH, 4, HEIGHT, WIDTH)


