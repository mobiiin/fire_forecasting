"""Alignment tests for CAWFE-Latte multimodal token fusion."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.models.cawfe_latte import (  # noqa: E402
    CAWFELatte,
    FireQueryCrossAttentionFusion,
    MultimodalAlignment,
    TerrainFiLMConditioner,
    tokens_to_grid,
)


BATCH = 2
TIME = 5
DIM = 64
HEIGHT = 2
WIDTH = 3


def _features(time: int = TIME, height: int = HEIGHT, width: int = WIDTH) -> list[torch.Tensor]:
    return [torch.randn(BATCH, time, DIM, height, width) for _ in range(4)]


def test_alignment_spatial_shape_validation_passes_and_fails() -> None:
    alignment = MultimodalAlignment(dim=DIM, max_time=TIME, use_spatial_pos=False, use_temporal_pos=False)
    out = alignment(*_features())
    assert tuple(out["atmosphere"].shape) == (BATCH, TIME, HEIGHT * WIDTH, DIM)
    bad = _features()
    bad[-1] = torch.randn(BATCH, TIME, DIM, HEIGHT + 1, WIDTH)
    with pytest.raises(ValueError, match="same spatial grid"):
        alignment(*bad)


def test_alignment_flattening_order_is_row_major() -> None:
    alignment = MultimodalAlignment(dim=1, max_time=1, use_spatial_pos=False, use_temporal_pos=False)
    x = torch.tensor([[[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]]])
    tokens = alignment.to_tokens(x)
    assert tokens.shape == (1, 1, 6, 1)
    assert tokens[0, 0, :, 0].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


def test_alignment_has_separate_layernorm_modules() -> None:
    alignment = MultimodalAlignment(dim=DIM, max_time=TIME)
    assert alignment.norm_atm is not alignment.norm_wind
    assert alignment.norm_atm is not alignment.norm_fire
    assert alignment.norm_atm is not alignment.norm_flux


def test_positional_embeddings_are_shared_across_modalities() -> None:
    alignment = MultimodalAlignment(dim=4, max_time=2, separate_layernorms=False)
    x = torch.ones(1, 2, 4, 2, 2)
    out = alignment(x, x, x, x)
    torch.testing.assert_close(out["atmosphere"], out["wind"])
    torch.testing.assert_close(out["atmosphere"], out["fire_fuel"])
    torch.testing.assert_close(out["atmosphere"], out["flux_energy"])
    assert alignment.spatial_pos is not None
    assert alignment.temporal_pos is not None


def test_temporal_alignment_max_time_validation() -> None:
    alignment = MultimodalAlignment(dim=DIM, max_time=5)
    alignment(*_features(time=5))
    alignment(*_features(time=1))
    with pytest.raises(ValueError, match="max_time=5"):
        alignment(*_features(time=6))


def test_token_fusion_output_and_attention_shapes() -> None:
    alignment = MultimodalAlignment(dim=DIM, max_time=TIME)
    aligned = alignment(*_features())
    fusion = FireQueryCrossAttentionFusion(dim=DIM, num_heads=4, dropout=0.0)
    z, weights = fusion(aligned["atmosphere"], aligned["wind"], aligned["fire_fuel"], aligned["flux_energy"], return_attention=True)
    assert tuple(z.shape) == (BATCH, TIME, HEIGHT * WIDTH, DIM)
    assert tuple(weights.shape) == (BATCH, TIME, HEIGHT * WIDTH, 3)


def test_tokens_to_grid_restores_row_major_tokens() -> None:
    tokens = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1)
    grid = tokens_to_grid(tokens, (2, 3))
    assert tuple(grid.shape) == (1, 1, 1, 2, 3)
    torch.testing.assert_close(grid[0, 0, 0], torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]))


def test_terrain_film_grid_validation_passes_and_fails() -> None:
    film = TerrainFiLMConditioner(dim=DIM)
    z = torch.randn(BATCH, TIME, DIM, HEIGHT, WIDTH)
    terrain = torch.randn(BATCH, DIM, HEIGHT, WIDTH)
    assert tuple(film(z, terrain).shape) == tuple(z.shape)
    with pytest.raises(ValueError, match="Terrain FiLM spatial/channel dimensions"):
        film(z, torch.randn(BATCH, DIM, HEIGHT + 1, WIDTH))


def test_full_cawfe_latte_forward_with_terrain() -> None:
    model = CAWFELatte(input_channels=129, input_sequence_length=TIME, output_channels=4, use_terrain_conditioning=True)
    x = torch.randn(BATCH, TIME, 129, 16, 16)
    terrain = torch.randn(BATCH, 4, 16, 16)
    out = model(x, terrain=terrain, return_features=True, return_attention=True)
    assert tuple(out["prediction"].shape) == (BATCH, 4, 16, 16)
    assert tuple(out["fused_tokens"].shape) == (BATCH, TIME, 16 * 16, DIM)
    assert tuple(out["fused_grid"].shape) == (BATCH, TIME, DIM, 16, 16)
    assert tuple(out["fusion_attention"].shape) == (BATCH, TIME, 16 * 16, 3)
    assert tuple(out["terrain_features"].shape) == (BATCH, DIM, 16, 16)
    assert tuple(out["fused_after_terrain"].shape) == (BATCH, TIME, DIM, 16, 16)
