"""Targeted tests for the first CAWFE-Latte architecture ablation batch."""

from __future__ import annotations

import pytest
import yaml

torch = pytest.importorskip("torch")

from scripts.check_cawfe_latte_ablation_configs import compare_configs  # noqa: E402
from src.config import load_config  # noqa: E402
from src.models.architecture_registry import ARCHITECTURE_REGISTRY  # noqa: E402
from src.models.cawfe_latte import (  # noqa: E402
    CAWFELatte,
    MultiScaleContextBackbone,
    ResidualSpatiotemporalBackbone,
    ResidualSpatiotemporalBlock,
    TemporalAggregator,
    TemporalAttentionPooling,
    TemporalCNNBackbone,
    build_post_fusion_backbone,
    build_temporal_pooling,
)
from src.models.model_factory import build_model_from_config  # noqa: E402
from src.training.model_outputs import extract_prediction  # noqa: E402


CONFIGS = {
    "baseline": "configs/ablations/cawfe_latte_baseline.yaml",
    "A_resblocks": "configs/ablations/cawfe_latte_A_resblocks.yaml",
    "B_multiscale_context": "configs/ablations/cawfe_latte_B_multiscale_context.yaml",
    "C_temporal_attention": "configs/ablations/cawfe_latte_C_temporal_attention.yaml",
}


def tiny_config(backbone_type: str, pooling_type: str) -> dict:
    return {
        "model": {"architecture": "cawfe_latte", "input_channels": 86, "output_channels": 4},
        "input_sequence_length": 2,
        "cawfe_latte": {
            "input_sequence_length": 2,
            "output_channels": 4,
            "output_dim": 8,
            "atmosphere": {"out_dim": 8, "hidden_dim": 8, "num_blocks": 0},
            "wind": {"out_dim": 8, "hidden_dim": 8, "num_blocks": 0},
            "fire_fuel": {"out_dim": 8, "hidden_dim": 8, "num_blocks": 0},
            "flux_energy": {"out_dim": 8, "hidden_dim": 8, "num_blocks": 0},
            "alignment": {"temporal": {"max_time": 2}},
            "fusion": {"dim": 8, "num_heads": 2, "dropout": 0.0},
            "backbone": {"dim": 8, "num_blocks": 1, "dropout": 0.0},
            "post_fusion_backbone": {
                "type": backbone_type,
                **({"num_blocks": 6} if backbone_type == "residual_spatiotemporal" else {}),
            },
            "temporal_aggregation": {"mode": "last"},
            "temporal_pooling": {"type": pooling_type},
            "decoder": {"in_dim": 8, "hidden_dim": 8, "num_blocks": 0, "dropout": 0.0},
            "auxiliary": {"fire_support_head": {"enabled": False}},
        },
    }


def test_baseline_config_instantiates_preserved_cawfe_latte() -> None:
    config = load_config(CONFIGS["baseline"])
    model = build_model_from_config(config, input_channels=129)
    assert isinstance(model, CAWFELatte)
    assert isinstance(model.post_fusion_backbone, TemporalCNNBackbone)
    assert len(model.post_fusion_backbone.blocks) == 3
    assert isinstance(model.temporal_pooling, TemporalAggregator)
    assert model.temporal_pooling.mode == "last"


def test_resblocks_preserve_shape_and_use_exactly_six_blocks() -> None:
    module = build_post_fusion_backbone(
        {"type": "residual_spatiotemporal", "num_blocks": 6, "dropout": 0.0},
        dim=8,
    )
    assert isinstance(module, ResidualSpatiotemporalBackbone)
    assert len(module.blocks) == 6
    assert all(isinstance(block, ResidualSpatiotemporalBlock) for block in module.blocks)
    x = torch.randn(2, 3, 8, 7, 9)
    assert module(x).shape == x.shape


def test_multiscale_context_preserves_shape_and_has_dilations_1_2_4() -> None:
    module = build_post_fusion_backbone({"type": "multiscale_context"}, dim=8)
    assert isinstance(module, MultiScaleContextBackbone)
    assert module.dilations == (1, 2, 4)
    assert tuple(branch[0].dilation for branch in module.branches) == ((1, 1), (2, 2), (4, 4))
    x = torch.randn(2, 3, 8, 7, 9)
    assert module(x).shape == x.shape


def test_temporal_attention_contract_and_uniform_initialization() -> None:
    pool = build_temporal_pooling({"type": "attention"}, dim=8, input_sequence_length=3)
    assert isinstance(pool, TemporalAttentionPooling)
    x = torch.randn(2, 3, 8, 7, 9)
    output = pool(x)
    alpha = pool.last_attention_weights
    assert output.shape == (2, 8, 7, 9)
    assert alpha is not None and alpha.shape == (2, 3, 1, 7, 9)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(2, 1, 7, 9))
    assert torch.allclose(alpha, torch.full_like(alpha, 1.0 / 3.0), atol=1.0e-7)


@pytest.mark.parametrize(
    ("backbone_type", "pooling_type"),
    [
        ("baseline_cnn", "baseline"),
        ("residual_spatiotemporal", "baseline"),
        ("multiscale_context", "baseline"),
        ("baseline_cnn", "attention"),
    ],
)
def test_all_first_batch_full_forwards_return_four_heads(backbone_type: str, pooling_type: str) -> None:
    model = build_model_from_config(tiny_config(backbone_type, pooling_type), input_channels=86).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 2, 86, 4, 4), return_features=pooling_type == "attention")
    assert extract_prediction(output).shape == (1, 4, 4, 4)
    if pooling_type == "attention":
        assert output["temporal_attention_alpha"].shape == (1, 2, 1, 4, 4)


def test_first_batch_configs_and_registry_are_narrow() -> None:
    registry = yaml.safe_load(open("configs/ablations/cawfe_latte_ablations.yaml", encoding="utf-8"))["ablations"]
    assert set(CONFIGS) <= set(registry)
    for name, path in CONFIGS.items():
        config = load_config(path)
        assert config["model"]["architecture"] == "cawfe_latte"
        assert config["cawfe_latte"]["ablation"]["name"] == name
        assert config["training"]["max_epochs"] == 10
        assert config["training"]["early_stopping"]["enabled"] is False
        assert config["training"]["run_test_after_training"] is False
        assert config["training"]["run_external_test_after_training"] is False
    assert compare_configs() == []
    assert not {"cawfe_latte_v1_1", "cawfe_latte_v1_2", "cawfe_latte_v1_3"} & set(ARCHITECTURE_REGISTRY)
