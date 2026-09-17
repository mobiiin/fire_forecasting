"""Targeted tests for the second CAWFE-Latte architecture ablation batch."""

from __future__ import annotations

import pytest
import yaml

torch = pytest.importorskip("torch")

from scripts.check_cawfe_latte_ablation_configs import compare_configs  # noqa: E402
from src.config import load_config  # noqa: E402
from src.models.cawfe_latte import (  # noqa: E402
    CAWFELatte,
    ConcatProjectionFusion,
    FireQueryCrossAttentionFusion,
    PatchFirePresenceHead,
)
from src.models.model_factory import build_model_from_config  # noqa: E402
from src.training.losses import MultiTaskLoss  # noqa: E402
from src.training.metrics import compute_patch_fire_metrics  # noqa: E402
from src.training.model_outputs import extract_prediction, patch_fire_presence_target  # noqa: E402


CONFIGS = {
    "G_separate_regression_decoder": "configs/ablations/cawfe_latte_G_separate_regression_decoder.yaml",
    "I_patch_fire_classifier": "configs/ablations/cawfe_latte_I_patch_fire_classifier.yaml",
    "K_no_terrain_film": "configs/ablations/cawfe_latte_K_no_terrain_film.yaml",
    "L_simple_concat_fusion": "configs/ablations/cawfe_latte_L_simple_concat_fusion.yaml",
}


def tiny_config(
    *,
    decoder_type: str = "shared",
    patch_fire: bool = False,
    terrain_film: bool = True,
    fusion_type: str = "fire_query_attention",
) -> dict:
    return {
        "task_type": "multitask",
        "model": {"architecture": "cawfe_latte", "input_channels": 86, "output_channels": 4},
        "input_sequence_length": 2,
        "energy_release": {"enabled": True, "output_mode": "total", "target_transform": "log1p"},
        "dataloader": {"source": "processed_full_frames"},
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
            "fusion": {"type": fusion_type, "dim": 8, "num_heads": 2, "dropout": 0.0},
            "terrain_encoder": {"in_channels": 4, "hidden_dim": 4, "out_dim": 8},
            "terrain_film": {"enabled": terrain_film, "dim": 8},
            "backbone": {"dim": 8, "num_blocks": 1, "dropout": 0.0},
            "post_fusion_backbone": {"type": "baseline_cnn"},
            "temporal_aggregation": {"mode": "last"},
            "temporal_pooling": {"type": "baseline"},
            "decoder": {
                "type": decoder_type,
                "in_dim": 8,
                "hidden_dim": 8,
                "num_blocks": 0,
                "dropout": 0.0,
            },
            "patch_fire_head": {"enabled": patch_fire, "hidden_dim": 4, "dropout": 0.0},
            "auxiliary": {"fire_support_head": {"enabled": False}},
        },
        "training": {
            "task_type": "multitask",
            "loss": {
                "surface": {"type": "huber", "weight": 1.0, "delta": 1.0},
                "canopy": {"type": "huber", "weight": 1.0, "delta": 1.0},
                "mask": {"type": "bce_dice", "weight": 5.0, "bce_weight": 1.0, "dice_weight": 1.0},
                "energy": {"type": "huber", "weight": 1.0, "delta": 1.0},
                "auxiliary_fire_support": {"enabled": False, "weight": 0.2},
                "patch_fire": {"enabled": patch_fire, "weight": 0.2},
            },
        },
    }


def sample_inputs(batch: int = 2):
    return torch.randn(batch, 2, 86, 5, 6), torch.randn(batch, 4, 5, 6)


def sample_targets(batch: int = 2):
    target = torch.rand(batch, 4, 5, 6)
    target[:, 2] = 0.0
    target[-1, 2, 1, 1] = 1.0
    return target


def test_g_shared_baseline_and_separate_decoder_routing() -> None:
    shared = build_model_from_config(tiny_config(), 86)
    assert isinstance(shared, CAWFELatte)
    assert shared.decoder is not None
    assert shared.mask_decoder is None and shared.regression_decoder is None

    model = build_model_from_config(tiny_config(decoder_type="separate_regression"), 86).eval()
    assert model.decoder is None
    assert model.mask_decoder is not None and model.regression_decoder is not None
    assert model.mask_decoder is not model.regression_decoder
    x, terrain = sample_inputs(1)
    with torch.no_grad():
        features = model(x, terrain=terrain, return_features=True)
    prediction = features["prediction"]
    assert prediction.shape == (1, 4, 5, 6)
    assert torch.allclose(prediction[:, 2:3], model.mask_head(features["mask_features"]))
    assert torch.allclose(prediction[:, 0:1], model.surface_head(features["regression_features"]))
    assert torch.allclose(prediction[:, 1:2], model.canopy_head(features["regression_features"]))
    assert torch.allclose(prediction[:, 3:4], model.energy_head(features["regression_features"]))


def test_i_patch_fire_head_target_loss_metrics_and_disabled_baseline() -> None:
    target = sample_targets()
    presence = patch_fire_presence_target(target)
    assert presence.shape == (2, 1)
    assert presence[:, 0].tolist() == [0.0, 1.0]

    model = build_model_from_config(tiny_config(patch_fire=True), 86).eval()
    assert isinstance(model.patch_fire_head, PatchFirePresenceHead)
    x, terrain = sample_inputs()
    output = model(x, terrain=terrain)
    assert output["patch_fire_logit"].shape == (2, 1)
    losses = MultiTaskLoss(tiny_config(patch_fire=True))(output, target)
    assert torch.isfinite(losses["loss_patch_fire_bce"])
    assert torch.isfinite(losses["total_loss"])
    metrics = compute_patch_fire_metrics(output, target)
    assert set(metrics) == {
        "patch_fire_accuracy",
        "patch_fire_precision",
        "patch_fire_recall",
        "patch_fire_f1",
    }

    baseline = build_model_from_config(tiny_config(patch_fire=False), 86).eval()
    baseline_output = baseline(x, terrain=terrain)
    assert baseline.patch_fire_head is None
    assert "patch_fire_logit" not in baseline_output if isinstance(baseline_output, dict) else True
    baseline_losses = MultiTaskLoss(tiny_config(patch_fire=False))(baseline_output, target)
    assert "loss_patch_fire_bce" not in baseline_losses


def test_k_terrain_film_disabled_bypasses_terrain_without_changing_shape() -> None:
    baseline = build_model_from_config(tiny_config(terrain_film=True), 86).eval()
    assert baseline.terrain_film_enabled
    assert baseline.terrain_encoder is not None and baseline.terrain_film is not None
    x, terrain = sample_inputs(1)
    with torch.no_grad():
        baseline_prediction = extract_prediction(baseline(x, terrain=terrain))
    assert baseline_prediction.shape == (1, 4, 5, 6)

    model = build_model_from_config(tiny_config(terrain_film=False), 86).eval()
    assert not model.terrain_film_enabled
    assert model.terrain_encoder is None and model.terrain_film is None
    with torch.no_grad():
        first = model(x, terrain=terrain, return_features=True)
        second = model(x, terrain=terrain + 100.0, return_features=True)
    assert first["prediction"].shape == (1, 4, 5, 6)
    assert torch.equal(first["prediction"], second["prediction"])
    assert torch.equal(first["fused_after_terrain"], first["fused_dynamic"])


def test_l_concat_projection_shape_and_no_attention_module() -> None:
    fusion = ConcatProjectionFusion(dim=8)
    inputs = [torch.randn(2, 3, 8, 5, 6) for _ in range(4)]
    assert fusion(*inputs).shape == (2, 3, 8, 5, 6)
    assert not any(isinstance(module, torch.nn.MultiheadAttention) for module in fusion.modules())

    baseline = build_model_from_config(tiny_config(), 86)
    assert isinstance(baseline.fusion, FireQueryCrossAttentionFusion)
    model = build_model_from_config(tiny_config(fusion_type="concat_projection"), 86).eval()
    assert isinstance(model.fusion, ConcatProjectionFusion)
    assert not any(isinstance(module, torch.nn.MultiheadAttention) for module in model.fusion.modules())
    x, terrain = sample_inputs(1)
    with torch.no_grad():
        prediction = extract_prediction(model(x, terrain=terrain))
    assert prediction.shape == (1, 4, 5, 6)


def test_batch2_configs_use_one_architecture_and_preserve_screening_protocol() -> None:
    registry = yaml.safe_load(open("configs/ablations/cawfe_latte_ablations.yaml", encoding="utf-8"))["ablations"]
    assert set(CONFIGS) <= set(registry)
    for name, path in CONFIGS.items():
        config = load_config(path)
        assert config["model"]["architecture"] == "cawfe_latte"
        assert config["cawfe_latte"]["ablation"]["name"] == name
        assert config["cawfe_latte"]["post_fusion_backbone"]["type"] == "baseline_cnn"
        assert config["cawfe_latte"]["temporal_pooling"]["type"] == "baseline"
        assert config["training"]["max_epochs"] == 10
        assert config["training"]["early_stopping"]["enabled"] is False
        assert config["training"]["run_test_after_training"] is False
        assert config["training"]["run_external_test_after_training"] is False
    assert compare_configs() == []
