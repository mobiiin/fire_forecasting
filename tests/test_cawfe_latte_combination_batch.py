"""Configuration, forward, loss, and summary tests for CAWFE-Latte combinations."""

from __future__ import annotations

import copy
import importlib.util
import json

import pytest
import yaml

torch = pytest.importorskip("torch")

from scripts.check_cawfe_latte_ablation_configs import compare_configs
from scripts.summarize_cawfe_latte_ablations import (
    add_baseline_deltas,
    add_parent_deltas,
    combination_synergy_lines,
    write_outputs,
)
from src.config import load_config
from src.models.cawfe_latte import (
    CuboidAttentionLiteBackbone,
    ResidualSpatiotemporalBackbone,
    SpatiotemporalMambaBackbone,
    TemporalAggregator,
    TemporalAttentionPooling,
    TemporalCNNBackbone,
)
from src.models.model_factory import build_model_from_config
from src.training.losses import MultiTaskLoss
from src.training.model_outputs import extract_prediction


COMBINATIONS = {
    "CG_separate_decoder_temporal_attention": ("baseline_cnn", True, True),
    "GA_separate_decoder_resblocks": ("residual_spatiotemporal", False, True),
    "GE_separate_decoder_earthformer": ("cuboid_attention_lite", False, True),
    "GP_separate_decoder_mamba": ("spatiotemporal_mamba", False, True),
    "GK_separate_decoder_no_terrain": ("baseline_cnn", False, False),
    "CGA_separate_decoder_temporal_attention_resblocks": ("residual_spatiotemporal", True, True),
    "CGE_separate_decoder_temporal_attention_earthformer": ("cuboid_attention_lite", True, True),
    "CGP_separate_decoder_temporal_attention_mamba": ("spatiotemporal_mamba", True, True),
    "CGK_separate_decoder_temporal_attention_no_terrain": ("baseline_cnn", True, False),
}


def config_path(name: str) -> str:
    return f"configs/ablations/cawfe_latte_{name}.yaml"


def tiny_config(name: str) -> dict:
    config = copy.deepcopy(load_config(config_path(name)))
    config["model"].update({"architecture": "cawfe_latte", "input_channels": 86, "output_channels": 4})
    config["input_sequence_length"] = 2
    section = config["cawfe_latte"]
    section.update({"input_sequence_length": 2, "output_channels": 4, "output_dim": 8})
    for encoder in ("atmosphere", "wind", "fire_fuel", "flux_energy"):
        section[encoder].update({"out_dim": 8, "hidden_dim": 8, "num_blocks": 0})
    section["alignment"]["temporal"]["max_time"] = 2
    section["fusion"].update({"dim": 8, "num_heads": 2, "dropout": 0.0})
    section["terrain_encoder"].update({"in_channels": 4, "hidden_dim": 4, "out_dim": 8})
    section["terrain_film"]["dim"] = 8
    section["backbone"].update({"dim": 8, "num_blocks": 1, "dropout": 0.0})
    post = section["post_fusion_backbone"]
    if "dim" in post:
        post["dim"] = 8
    if "num_heads" in post:
        post["num_heads"] = 2
    if post["type"] == "cuboid_attention_lite":
        post["cuboid_size"] = [2, 4, 4]
    post["dropout"] = 0.0
    section["decoder"].update({"in_dim": 8, "hidden_dim": 8, "num_blocks": 0, "dropout": 0.0})
    section["auxiliary"]["fire_support_head"]["enabled"] = False
    return config


def test_combination_configs_are_registered_narrow_and_keep_protocol() -> None:
    registry = yaml.safe_load(open("configs/ablations/cawfe_latte_ablations.yaml", encoding="utf-8"))["ablations"]
    assert set(COMBINATIONS) <= set(registry)
    for name in COMBINATIONS:
        config = load_config(config_path(name))
        entry = registry[name]
        assert config["model"]["architecture"] == "cawfe_latte"
        assert config["cawfe_latte"]["ablation"]["name"] == name
        assert entry["name"] == name
        assert entry["short_name"]
        assert len(entry["components"]) in (2, 3)
        assert entry["changed_components"]
        assert entry["expected_compute_class"] in {"medium", "high"}
        assert config["training"]["max_epochs"] == 10
        assert config["training"]["early_stopping"]["enabled"] is False
        assert config["training"]["run_test_after_training"] is False
        assert config["training"]["run_external_test_after_training"] is False
    assert compare_configs() == []


@pytest.mark.parametrize(("name", "backbone_type", "attention", "terrain"), [(name, *settings) for name, settings in COMBINATIONS.items()])
def test_combination_modules_forward_and_finite_loss(name: str, backbone_type: str, attention: bool, terrain: bool) -> None:
    if backbone_type == "spatiotemporal_mamba" and importlib.util.find_spec("mamba_ssm") is None:
        pytest.skip("official mamba_ssm dependency unavailable")
    config = tiny_config(name)
    model = build_model_from_config(config, 86)
    expected_backbone = {
        "baseline_cnn": TemporalCNNBackbone,
        "residual_spatiotemporal": ResidualSpatiotemporalBackbone,
        "cuboid_attention_lite": CuboidAttentionLiteBackbone,
        "spatiotemporal_mamba": SpatiotemporalMambaBackbone,
    }[backbone_type]
    assert isinstance(model.post_fusion_backbone, expected_backbone)
    assert isinstance(model.temporal_pooling, TemporalAttentionPooling if attention else TemporalAggregator)
    assert model.temporal_pooling.mode == "last" if isinstance(model.temporal_pooling, TemporalAggregator) else True
    assert model.decoder is None and model.mask_decoder is not None and model.regression_decoder is not None
    assert model.terrain_film_enabled is terrain
    if backbone_type == "spatiotemporal_mamba" and not torch.cuda.is_available():
        pytest.skip("official mamba_ssm runtime requires CUDA")
    device = torch.device("cuda" if backbone_type == "spatiotemporal_mamba" else "cpu")
    model = model.to(device).eval()
    x = torch.randn(1, 2, 86, 4, 4, device=device)
    terrain_input = torch.randn(1, 4, 4, 4, device=device)
    target = torch.rand(1, 4, 4, 4, device=device)
    target[:, 2] = (target[:, 2] > 0.5).float()
    output = model(x, terrain=terrain_input)
    prediction = extract_prediction(output)
    assert prediction.shape == (1, 4, 4, 4)
    loss = MultiTaskLoss(config)(output, target)["total_loss"]
    assert torch.isfinite(loss)


def test_parent_deltas_and_synergy_are_generated_mechanically() -> None:
    metrics = {
        "Val Dice": 0.5, "Val IoU": 0.4, "Val Energy Log MAE": 0.4,
        "Val Surface MAE": 0.3, "Val Canopy MAE": 0.2, "Val Active Canopy MAE": 0.5,
    }
    rows = [
        {"Ablation": "baseline", "_mtime": 1.0, "_components": [], **metrics},
        {"Ablation": "C_temporal_attention", "_mtime": 1.0, "_components": ["C"], **{**metrics, "Val Dice": 0.6}},
        {"Ablation": "G_separate_regression_decoder", "_mtime": 1.0, "_components": ["G"], **{**metrics, "Val Surface MAE": 0.2}},
        {"Ablation": "CG_separate_decoder_temporal_attention", "_short_name": "CG", "_mtime": 2.0, "_components": ["C", "G"], **{**metrics, "Val Dice": 0.7, "Val Surface MAE": 0.1}},
    ]
    add_baseline_deltas(rows)
    add_parent_deltas(rows)
    cg = rows[-1]
    assert cg["Delta Val Dice"] == pytest.approx(0.2)
    assert cg["Delta vs C Val Dice"] == pytest.approx(0.1)
    assert cg["Delta vs G Val Surface MAE"] == pytest.approx(-0.1)
    assert set(cg["_parent_deltas"]) == {"baseline", "C", "G"}
    synergy = "\n".join(combination_synergy_lines(rows))
    assert "CG:" in synergy and "vs baseline:" in synergy and "improved: Dice, Surface MAE" in synergy


def test_epoch_metric_logging_keeps_finite_no_fire_batches(monkeypatch) -> None:
    import src.training.train as train_module

    calls = iter([
        {"no_fire_patch_count": 2.0, "no_fire_mask_false_positive_rate": 0.25, "active_energy_log_mae": float("nan")},
        {"no_fire_patch_count": 0.0, "no_fire_mask_false_positive_rate": float("nan"), "active_energy_log_mae": 0.3},
    ])
    monkeypatch.setattr(train_module, "compute_metrics", lambda *_args, **_kwargs: next(calls))

    class Model(torch.nn.Module):
        def forward(self, x):
            return x[:, -1, :1]

    loader = [(torch.ones(2, 2, 1, 2, 2), torch.zeros(2, 1, 2, 2)) for _ in range(2)]
    result = train_module._run_epoch(
        model=Model(), loader=loader, criterion=torch.nn.MSELoss(),
        config={"training": {"performance": {"compute_val_metrics": True, "show_progress_bar": False}}},
        device=torch.device("cpu"), input_sequence_length=2, input_channels=1,
        output_channels=1, train=False,
    )
    assert result["val_no_fire_patch_count"] == 2.0
    assert result["val_no_fire_mask_false_positive_rate"] == pytest.approx(0.25)
    assert result["val_active_energy_log_mae"] == pytest.approx(0.3)


def test_combination_parent_deltas_reach_csv_json_and_synergy_outputs(tmp_path) -> None:
    metrics = {
        "Val Dice": 0.5, "Val IoU": 0.4, "Val Energy Log MAE": 0.4,
        "Val Surface MAE": 0.3, "Val Canopy MAE": 0.2, "Val Active Canopy MAE": 0.5,
    }
    rows = [
        {"Ablation": "baseline", "Components": "baseline", "_mtime": 1.0, "_components": [], **metrics},
        {"Ablation": "C_temporal_attention", "Components": "C", "_mtime": 1.0, "_components": ["C"], **metrics},
        {"Ablation": "G_separate_regression_decoder", "Components": "G", "_mtime": 1.0, "_components": ["G"], **metrics},
        {"Ablation": "CG_separate_decoder_temporal_attention", "Components": "C, G", "_short_name": "CG", "_mtime": 2.0, "_components": ["C", "G"], **{**metrics, "Val Dice": 0.6}},
    ]
    add_baseline_deltas(rows)
    add_parent_deltas(rows)
    write_outputs(tmp_path, rows)
    assert "Delta vs C Val Dice" in (tmp_path / "ablation_results.csv").read_text()
    payload = json.loads((tmp_path / "ablation_results.json").read_text())
    assert payload[-1]["Parent Deltas"]["C"]["Val Dice"] == pytest.approx(0.1)
    report = (tmp_path / "ablation_results.txt").read_text()
    assert "BEST COMBINATIONS" in report and "Combination Synergy" in report and "vs G:" in report
