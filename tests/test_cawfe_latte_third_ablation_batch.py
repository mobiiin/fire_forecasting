"""Targeted tests for the third CAWFE-Latte architecture ablation batch."""

from __future__ import annotations

import importlib.util

import pytest
import yaml

torch = pytest.importorskip("torch")

from scripts.check_cawfe_latte_ablation_configs import compare_configs  # noqa: E402
from scripts.sanity_check_project import _resolve_sanity_device  # noqa: E402
from src.config import load_config  # noqa: E402
from src.models.cawfe_latte import (  # noqa: E402
    CuboidAttentionLiteBackbone,
    LocalWindowAttentionBackbone,
    SpatiotemporalMambaBlock,
    SpectralFourierBackbone,
    build_post_fusion_backbone,
    cuboid_partition,
    cuboid_reverse,
)
from src.models.model_factory import build_model_from_config  # noqa: E402
from src.training.model_outputs import extract_prediction  # noqa: E402


CONFIGS = {
    "D_local_window_attention": "configs/ablations/cawfe_latte_D_local_window_attention.yaml",
    "E_earthformer_lite": "configs/ablations/cawfe_latte_E_earthformer_lite.yaml",
    "O_fourier_postfusion": "configs/ablations/cawfe_latte_O_fourier_postfusion.yaml",
    "P_mamba_postfusion": "configs/ablations/cawfe_latte_P_mamba_postfusion.yaml",
}


def tiny_config(backbone: dict) -> dict:
    return {
        "model": {"architecture": "cawfe_latte", "input_channels": 86, "output_channels": 4},
        "input_sequence_length": 2,
        "cawfe_latte": {
            "input_sequence_length": 2,
            "output_channels": 4,
            "output_dim": 8,
            "use_terrain_conditioning": True,
            "atmosphere": {"out_dim": 8, "hidden_dim": 8, "num_blocks": 0},
            "wind": {"out_dim": 8, "hidden_dim": 8, "num_blocks": 0},
            "fire_fuel": {"out_dim": 8, "hidden_dim": 8, "num_blocks": 0},
            "flux_energy": {"out_dim": 8, "hidden_dim": 8, "num_blocks": 0},
            "alignment": {"temporal": {"max_time": 2}},
            "fusion": {"type": "fire_query_attention", "dim": 8, "num_heads": 2, "dropout": 0.0},
            "terrain_encoder": {"in_channels": 4, "hidden_dim": 4, "out_dim": 8},
            "terrain_film": {"enabled": True, "dim": 8},
            "backbone": {"dim": 8, "num_blocks": 1, "dropout": 0.0},
            "post_fusion_backbone": backbone,
            "temporal_aggregation": {"mode": "last"},
            "temporal_pooling": {"type": "baseline"},
            "decoder": {"type": "shared", "in_dim": 8, "hidden_dim": 8, "num_blocks": 0, "dropout": 0.0},
            "auxiliary": {"fire_support_head": {"enabled": False}},
        },
    }


def full_forward(backbone: dict, *, height: int = 5, width: int = 6):
    model = build_model_from_config(tiny_config(backbone), 86).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 2, 86, height, width), terrain=torch.randn(1, 4, height, width), return_features=True)
    assert extract_prediction(output).shape == (1, 4, height, width)
    return model, output


def test_d_local_windows_preserve_divisible_and_padded_shapes_and_debug_info() -> None:
    module = build_post_fusion_backbone({"type": "local_window_attention", "dim": 8, "num_blocks": 2, "num_heads": 2, "window_size": 4, "mlp_ratio": 2.0, "dropout": 0.0}, dim=8)
    assert isinstance(module, LocalWindowAttentionBackbone)
    for shape in ((2, 3, 8, 8, 12), (2, 3, 8, 7, 9)):
        x = torch.randn(*shape)
        assert module(x).shape == x.shape
    _, features = full_forward({"type": "local_window_attention", "dim": 8, "num_blocks": 2, "num_heads": 2, "window_size": 4, "mlp_ratio": 2.0, "dropout": 0.0})
    assert features["local_window_attention_enabled"] is True
    assert features["window_size"] == 4


def test_e_cuboid_partition_reverse_shift_and_full_forward() -> None:
    grid = torch.arange(2 * 4 * 8 * 12 * 3).reshape(2, 4, 8, 12, 3)
    windows = cuboid_partition(grid, (2, 4, 4))
    reconstructed = cuboid_reverse(windows, (2, 4, 4), 2, 4, 8, 12)
    assert torch.equal(reconstructed, grid)
    module = build_post_fusion_backbone({"type": "cuboid_attention_lite", "dim": 8, "num_blocks": 2, "num_heads": 2, "cuboid_size": [4, 4, 4], "use_shifted_cuboids": True, "mlp_ratio": 2.0, "dropout": 0.0}, dim=8)
    assert isinstance(module, CuboidAttentionLiteBackbone)
    assert module.blocks[0].shifted is False and module.blocks[1].shifted is True
    x = torch.randn(1, 2, 8, 7, 9)
    assert module(x).shape == x.shape
    full_forward({"type": "cuboid_attention_lite", "dim": 8, "num_blocks": 2, "num_heads": 2, "cuboid_size": [2, 4, 4], "use_shifted_cuboids": True, "mlp_ratio": 2.0, "dropout": 0.0})


def test_o_fourier_cpu_finite_gradient_and_full_forward() -> None:
    module = build_post_fusion_backbone({"type": "spectral_fourier", "dim": 8, "num_blocks": 2, "modes_h": 4, "modes_w": 4, "residual": True, "dropout": 0.0}, dim=8)
    assert isinstance(module, SpectralFourierBackbone)
    x = torch.randn(2, 3, 8, 7, 9, requires_grad=True)
    output = module(x)
    assert output.shape == x.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert module.blocks[0].spectral.weight.grad is not None
    full_forward({"type": "spectral_fourier", "dim": 8, "num_blocks": 2, "modes_h": 4, "modes_w": 4, "residual": True, "dropout": 0.0})


def test_p_temporal_and_spatial_reshape_roundtrips() -> None:
    x = torch.randn(2, 3, 8, 5, 7)
    temporal, shape = SpatiotemporalMambaBlock.to_temporal_sequences(x)
    assert temporal.shape == (2 * 5 * 7, 3, 8)
    assert torch.equal(SpatiotemporalMambaBlock.from_temporal_sequences(temporal, shape), x)
    spatial, shape = SpatiotemporalMambaBlock.to_spatial_sequences(x)
    assert spatial.shape == (2 * 3, 5 * 7, 8)
    assert torch.equal(SpatiotemporalMambaBlock.from_spatial_sequences(spatial, shape), x)


@pytest.mark.skipif(importlib.util.find_spec("mamba_ssm") is None or not torch.cuda.is_available(), reason="official mamba_ssm CUDA runtime unavailable")
def test_p_official_mamba_shape_gradient_and_full_forward() -> None:
    backbone = {"type": "spatiotemporal_mamba", "dim": 8, "num_blocks": 2, "expansion": 2, "dropout": 0.0, "scan_order": "temporal_then_spatial", "backend": "mamba_ssm"}
    module = build_post_fusion_backbone(backbone, dim=8).cuda()
    x = torch.randn(1, 2, 8, 3, 3, device="cuda", requires_grad=True)
    output = module(x)
    assert output.shape == x.shape and torch.isfinite(output).all()
    output.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    model = build_model_from_config(tiny_config(backbone), 86).cuda().eval()
    result = model(torch.randn(1, 2, 86, 3, 3, device="cuda"), terrain=torch.randn(1, 4, 3, 3, device="cuda"))
    assert extract_prediction(result).shape == (1, 4, 3, 3)


def test_p_sanity_check_resolves_cuda_and_honors_override(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert str(_resolve_sanity_device({"device": "cuda"})) == "cuda"
    assert str(_resolve_sanity_device({"device": "cuda"}, "cpu")) == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(_resolve_sanity_device({"device": "cuda"})) == "cpu"


def test_batch3_configs_registry_protocol_and_compute_classes() -> None:
    registry = yaml.safe_load(open("configs/ablations/cawfe_latte_ablations.yaml", encoding="utf-8"))["ablations"]
    expected_compute = {"D_local_window_attention": "medium", "E_earthformer_lite": "high", "O_fourier_postfusion": "medium", "P_mamba_postfusion": "high"}
    assert set(CONFIGS) <= set(registry)
    for name, path in CONFIGS.items():
        config = load_config(path)
        assert config["model"]["architecture"] == "cawfe_latte"
        assert config["cawfe_latte"]["ablation"]["name"] == name
        assert config["cawfe_latte"]["temporal_pooling"]["type"] == "baseline"
        assert config["cawfe_latte"]["terrain_film"]["enabled"] is True
        assert config["cawfe_latte"]["decoder"]["type"] == "shared"
        assert config["training"]["max_epochs"] == 10
        assert config["training"]["gradient_accumulation_steps"] == 1
        assert config["training"]["early_stopping"]["enabled"] is False
        assert config["training"]["run_test_after_training"] is False
        assert config["training"]["run_external_test_after_training"] is False
        assert registry[name]["changed_component"] == "post_fusion_backbone"
        assert registry[name]["expected_compute_class"] == expected_compute[name]
    assert compare_configs() == []
