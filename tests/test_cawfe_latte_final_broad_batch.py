"""Targeted tests for the final broad CAWFE-Latte screening batch."""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")

from scripts.check_cawfe_latte_ablation_configs import compare_configs
from scripts.check_patch_fire_balance import active_fraction_bin
from scripts.summarize_cawfe_latte_ablations import add_parent_deltas, pareto_candidates, text_footer
from src.config import load_config
from src.evaluation.fire_activity import active_fraction_bin_index, active_fraction_bin_name
from src.models.cawfe_latte import (
    ContrastiveProjectionHead,
    MaskGuidedRegressionAttention,
    RegressionMixtureOfExperts,
    ResidualSpatiotemporalBackbone,
    SpatiotemporalMambaBackbone,
    TemporalAggregator,
    TemporalAttentionPooling,
    gradient_reverse,
)
from src.models.model_factory import build_model_from_config
from src.training.losses import (
    MultiTaskLoss,
    active_fraction_class_labels,
    cross_fire_mmd,
    physical_state_targets,
    rbf_mmd,
    supervised_contrastive_loss,
    supervised_contrastive_positive_mask,
)
from src.training.model_outputs import extract_prediction


NEW_EXPERIMENTS = {
    "GA_Q1_fire_domain_adversarial": "GA-Q1",
    "GA_Q2_fire_domain_mmd": "GA-Q2",
    "GA_R_mask_guided_regression_attention": "GA-R",
    "GA_S_regression_moe": "GA-S",
    "GA_T_supervised_contrastive": "GA-T",
    "GA_U_physical_state_aux": "GA-U",
    "GPK_mamba_no_terrain": "GPK",
    "CGPK_temporal_mamba_no_terrain": "CGPK",
    "GAK_resblocks_no_terrain": "GAK",
    "CGP_R_mask_guided_regression_attention": "CGP-R",
    "GP_R_mask_guided_regression_attention": "GP-R",
    "GK_R_mask_guided_regression_attention": "GK-R",
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
    # Full-forward smoke tests need only one lightweight block; exact six/two
    # block settings are verified from the resolved screening configs below.
    post["num_blocks"] = 1
    post["dropout"] = 0.0
    section["decoder"].update({"in_dim": 8, "hidden_dim": 8, "num_blocks": 0, "dropout": 0.0})
    section["auxiliary"]["fire_support_head"]["enabled"] = False
    if "domain_adversarial" in section:
        section["domain_adversarial"]["num_domains"] = 3
        section["domain_adversarial"]["hidden_dim"] = 4
    if "supervised_contrastive" in section:
        section["supervised_contrastive"]["projection_dim"] = 4
    if "physical_state_aux" in section:
        section["physical_state_aux"]["hidden_dim"] = 4
    return config


def batch(device: torch.device = torch.device("cpu")):
    x = torch.randn(4, 2, 86, 4, 4, device=device)
    terrain = torch.randn(4, 4, 4, 4, device=device)
    target = torch.rand(4, 4, 4, 4, device=device)
    target[:, 2] = (target[:, 2] > 0.65).float()
    return x, terrain, target


def test_q1_grl_identity_and_exact_gradient_reversal() -> None:
    x = torch.tensor([1.0, -2.0], requires_grad=True)
    coefficients = torch.tensor([3.0, 5.0])
    output = gradient_reverse(x, 1.0)
    assert torch.equal(output, x)
    (output * coefficients).sum().backward()
    assert torch.allclose(x.grad, -coefficients)


def test_q1_domain_head_uses_train_ids_but_eval_needs_no_domain_id() -> None:
    config = tiny_config("GA_Q1_fire_domain_adversarial")
    model = build_model_from_config(config, 86)
    x, terrain, target = batch()
    model.train()
    output = model(x, terrain=terrain)
    assert output["domain_logits"].shape == (4, 3)
    criterion = MultiTaskLoss(config)
    forecast_only = criterion(dict(output), target)["total_loss"]
    output["auxiliary_training"] = True
    output["fire_domain_labels"] = torch.tensor([0, 1, 2, 1])
    training_loss = criterion(output, target)
    assert torch.isfinite(training_loss["total_loss"])
    assert torch.isfinite(training_loss["domain_loss"])
    assert training_loss["domain_random_chance"].item() == pytest.approx(1.0 / 3.0)
    assert (training_loss["total_loss"] - forecast_only).item() == pytest.approx(
        0.05 * training_loss["domain_loss"].item(), rel=1.0e-5
    )
    assert model.domain_adversarial_head.lambda_grl == 1.0
    assert isinstance(model.domain_adversarial_head.classifier[2], torch.nn.Dropout)
    assert model.domain_adversarial_head.classifier[2].p == pytest.approx(0.1)
    model.eval()
    with torch.no_grad():
        validation_output = model(x, terrain=terrain)
        validation_loss = MultiTaskLoss(config)(validation_output, target)
    assert not isinstance(validation_output, dict) or "domain_logits" not in validation_output
    assert torch.isfinite(validation_loss["total_loss"])



def test_q1_training_loop_maps_metadata_and_validation_ignores_unseen_fire() -> None:
    from src.training.train import _run_epoch

    config = tiny_config("GA_Q1_fire_domain_adversarial")
    config["_fire_domain_to_index"] = {"fire_a": 0, "fire_b": 1, "fire_c": 2}
    config["training"]["performance"].update({"show_progress_bar": False, "compute_val_metrics": True})
    model = build_model_from_config(config, 86)
    criterion = MultiTaskLoss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    x, terrain, target = batch()
    train_loader = [{
        "x": x,
        "y": target,
        "terrain": terrain,
        "metadata": {"fire_name": ["fire_a", "fire_b", "fire_c", "fire_b"]},
    }]
    train_result = _run_epoch(
        model=model, loader=train_loader, criterion=criterion, config=config,
        device=torch.device("cpu"), input_sequence_length=2, input_channels=86,
        output_channels=4, train=True, optimizer=optimizer,
    )
    assert torch.isfinite(torch.tensor(train_result["train_domain_loss"]))
    assert 0.0 <= train_result["train_domain_accuracy"] <= 1.0

    validation_loader = [{
        "x": x,
        "y": target,
        "terrain": terrain,
        "metadata": {"fire_name": ["unseen_1", "unseen_2", "unseen_3", "unseen_4"]},
    }]
    validation_result = _run_epoch(
        model=model, loader=validation_loader, criterion=criterion, config=config,
        device=torch.device("cpu"), input_sequence_length=2, input_channels=86,
        output_channels=4, train=False,
    )
    assert torch.isfinite(torch.tensor(validation_result["val_loss"]))
    assert "val_domain_loss" not in validation_result


def test_q1_mapping_is_sorted_training_only_and_saved(tmp_path) -> None:
    from src.training.train import _configure_fire_domain_training, build_fire_domain_mapping

    records = [{"fire_name": "zeta"}, {"fire_name": "alpha"}, {"fire_name": "zeta"}]
    assert build_fire_domain_mapping(records) == {"alpha": 0, "zeta": 1}
    config = tiny_config("GA_Q1_fire_domain_adversarial")
    config["cawfe_latte"]["domain_adversarial"]["num_domains"] = 2
    config["training"]["run_dir"] = str(tmp_path)
    loader = SimpleNamespace(dataset=SimpleNamespace(records=records), batch_size=2)
    _configure_fire_domain_training(config, loader, logging.getLogger(__name__))
    mapping_path = tmp_path / "train_fire_domain_mapping.json"
    assert json.loads(mapping_path.read_text(encoding="utf-8")) == {"alpha": 0, "zeta": 1}
    assert config["_fire_domain_mapping_path"] == str(mapping_path)


def test_q2_mmd_is_finite_zero_for_identical_and_safe_for_one_domain() -> None:
    features = torch.randn(6, 8, requires_grad=True)
    value = rbf_mmd(features, features, [0.5, 1.0, 2.0, 4.0])
    assert torch.isfinite(value) and value.item() == pytest.approx(0.0, abs=1.0e-7)
    single_loss, valid = cross_fire_mmd(features, torch.zeros(6, dtype=torch.long), [1.0])
    assert single_loss.item() == pytest.approx(0.0) and valid.item() == pytest.approx(0.0)
    mixed_loss, valid = cross_fire_mmd(features, torch.tensor([0, 0, 0, 1, 1, 1]), [0.5, 1.0])
    assert torch.isfinite(mixed_loss) and valid.item() == pytest.approx(1.0)



def test_q2_full_model_loss_is_finite_and_logs_contributing_batch() -> None:
    config = tiny_config("GA_Q2_fire_domain_mmd")
    model = build_model_from_config(config, 86).train()
    x, terrain, target = batch()
    output = model(x, terrain=terrain)
    assert output["fire_mmd_features"].shape == (4, 8)
    output["auxiliary_training"] = True
    output["fire_domain_labels"] = torch.tensor([0, 0, 1, 1])
    result = MultiTaskLoss(config)(output, target)
    assert torch.isfinite(result["total_loss"])
    assert torch.isfinite(result["mmd_loss"])
    assert result["mmd_valid_batch_fraction"].item() == pytest.approx(1.0)

def test_r_is_parent_equivalent_at_zero_and_detaches_mask_features() -> None:
    module = MaskGuidedRegressionAttention(mask_dim=8, alpha_init=0.0)
    regression = torch.randn(2, 8, 4, 4, requires_grad=True)
    mask = torch.randn(2, 8, 4, 4, requires_grad=True)
    enhanced, attention = module(regression, mask)
    assert module.alpha.item() == 0.0
    assert torch.equal(enhanced, regression)
    enhanced.sum().backward()
    assert module.alpha.grad is not None and torch.isfinite(module.alpha.grad)
    assert mask.grad is None

    module.zero_grad(set_to_none=True)
    regression_2 = torch.randn(2, 8, 4, 4, requires_grad=True)
    mask_2 = torch.randn(2, 8, 4, 4, requires_grad=True)
    module.alpha.data.fill_(0.25)
    module(regression_2, mask_2)[0].sum().backward()
    assert regression_2.grad is not None
    assert mask_2.grad is None
    assert module.projection.weight.grad is not None
    assert torch.isfinite(module.projection.weight.grad).all()
    assert attention.min() >= 0 and attention.max() <= 1




def test_r_full_regression_path_cannot_backpropagate_into_mask_decoder() -> None:
    model = build_model_from_config(tiny_config("GA_R_mask_guided_regression_attention"), 86).train()
    model.mask_guided_regression.alpha.data.fill_(0.25)
    x, terrain, _ = batch()
    features = model(x, terrain=terrain, return_features=True)
    features["regression_features"].square().mean().backward()
    assert model.mask_guided_regression.alpha.grad is not None
    assert model.mask_guided_regression.projection.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.mask_decoder.parameters())


def test_r_full_model_is_seed_for_seed_parent_equivalent_at_initialization() -> None:
    parent_config = tiny_config("GA_R_mask_guided_regression_attention")
    parent_config["cawfe_latte"].pop("mask_guided_regression")
    r_config = tiny_config("GA_R_mask_guided_regression_attention")
    torch.manual_seed(1234)
    parent = build_model_from_config(parent_config, 86).eval()
    torch.manual_seed(1234)
    guided = build_model_from_config(r_config, 86).eval()
    x, terrain, _ = batch()
    with torch.no_grad():
        torch.manual_seed(5678)  # lazy learned spatial-position initialization
        parent_features = parent(x, terrain=terrain, return_features=True)
        torch.manual_seed(5678)
        guided_features = guided(x, terrain=terrain, return_features=True)
    torch.testing.assert_close(guided_features["regression_features"], parent_features["regression_features"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(guided_features["prediction"], parent_features["prediction"], rtol=0.0, atol=0.0)
    assert guided_features["z_shared"].shape == (4, 8, 4, 4)

def test_s_experts_router_mixture_shared_heads_and_gradients() -> None:
    module = RegressionMixtureOfExperts(
        8, {"in_dim": 8, "hidden_dim": 8, "num_blocks": 0, "dropout": 0.0}, num_experts=3
    )
    latent = torch.randn(5, 8, 4, 4, requires_grad=True)
    features, weights = module(latent)
    assert features.shape == (5, 8, 4, 4)
    assert len(module.experts) == 3
    assert len({id(expert) for expert in module.experts}) == 3
    parameter_shapes = [[tuple(parameter.shape) for parameter in expert.parameters()] for expert in module.experts]
    assert parameter_shapes[0] == parameter_shapes[1] == parameter_shapes[2]
    for parameter_index in range(len(list(module.experts[0].parameters()))):
        pointers = {list(expert.parameters())[parameter_index].data_ptr() for expert in module.experts}
        assert len(pointers) == 3
    assert isinstance(module.router[0], torch.nn.Linear) and module.router[0].out_features == 4
    assert isinstance(module.router[1], torch.nn.SiLU)
    assert isinstance(module.router[2], torch.nn.Linear) and module.router[2].out_features == 3
    assert torch.all(weights > 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(5), atol=1.0e-7)
    (features.square().mean() + weights[:, 0].mean()).backward()
    assert all(any(parameter.grad is not None for parameter in expert.parameters()) for expert in module.experts)
    assert all(parameter.grad is not None for parameter in module.router.parameters())

    model = build_model_from_config(tiny_config("GA_S_regression_moe"), 86)
    assert model.regression_decoder is None
    assert model.surface_head is not None and model.canopy_head is not None and model.energy_head is not None
    assert not any(hasattr(expert, "surface_head") for expert in model.regression_moe.experts)
    x, terrain, target = batch()
    output = model.train()(x, terrain=terrain)
    output["auxiliary_training"] = True
    diagnostics = MultiTaskLoss(tiny_config("GA_S_regression_moe"))(output, target)
    for name in ("expert_1_mean_weight", "expert_2_mean_weight", "expert_3_mean_weight", "router_entropy", "router_max_probability_mean", "load_balance_loss"):
        assert torch.isfinite(diagnostics[name])


def test_t_activity_bins_positive_pairs_normalization_and_no_positive_batch() -> None:
    fractions = (0.0, 0.0005, 0.001, 0.01, 0.05)
    target = torch.zeros(5, 4, 100, 100)
    for row, active_count in enumerate((0, 5, 10, 100, 500)):
        target[row, 2].view(-1)[:active_count] = 1.0
    assert active_fraction_class_labels(target).tolist() == [0, 1, 2, 3, 4]
    assert [active_fraction_bin_index(value) for value in fractions] == [0, 1, 2, 3, 4]
    assert [active_fraction_bin(value) for value in fractions] == [active_fraction_bin_name(value) for value in fractions]

    positive_mask = supervised_contrastive_positive_mask(torch.tensor([0, 0, 1]))
    assert not positive_mask.diagonal().any()
    assert positive_mask[0, 1] and positive_mask[1, 0]
    assert not positive_mask[0, 2]

    head = ContrastiveProjectionHead(8)
    assert head.net[0].out_features == 128 and head.net[2].out_features == 64
    embeddings = head(torch.randn(5, 8))
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(5), atol=1.0e-6)
    loss, valid_fraction = supervised_contrastive_loss(embeddings, torch.arange(5), 0.1)
    assert torch.isfinite(loss) and loss.item() == pytest.approx(0.0)
    assert valid_fraction.item() == pytest.approx(0.0)
    one_class_loss, one_class_valid = supervised_contrastive_loss(embeddings, torch.zeros(5, dtype=torch.long), 0.1)
    assert torch.isfinite(one_class_loss) and one_class_valid.item() == pytest.approx(1.0)


def test_u_physical_targets_zero_no_fire_means_and_finite_auxiliary_loss() -> None:
    target = torch.zeros(2, 4, 2, 2)
    target[1, 2, 0, 0] = 1.0
    target[1, 1, 0, 0] = 0.4
    target[1, 3, 0, 0] = 2.0
    physical = physical_state_targets(target)
    assert torch.equal(physical[0], torch.zeros(3))
    assert torch.allclose(physical[1], torch.tensor([0.25, torch.log1p(torch.tensor(0.4)), 2.0]))
    assert physical[1, 2].item() == pytest.approx(2.0)

    config = tiny_config("GA_U_physical_state_aux")
    model = build_model_from_config(config, 86).eval()
    x, terrain, batch_target = batch()
    with torch.no_grad():
        prediction = model(x, terrain=terrain)
        output = model(x, terrain=terrain, return_aux=True)
    assert torch.is_tensor(prediction) and prediction.shape == (4, 4, 4, 4)
    torch.testing.assert_close(output["prediction"], prediction)
    assert torch.all((output["physical_state_prediction"][:, 0] >= 0) & (output["physical_state_prediction"][:, 0] <= 1))
    criterion = MultiTaskLoss(config)
    diagnostic_loss = criterion(output, batch_target)
    for name in ("physical_active_fraction_mae", "physical_canopy_state_mae", "physical_energy_state_mae"):
        assert torch.isfinite(diagnostic_loss[name])

    train_output = model.train()(x, terrain=terrain)
    forecast_only = criterion(dict(train_output), batch_target)["total_loss"]
    train_output["auxiliary_training"] = True
    combined = criterion(train_output, batch_target)
    assert (combined["total_loss"] - forecast_only).item() == pytest.approx(
        0.05 * combined["physical_state_aux_loss"].item(), rel=1.0e-4, abs=2.0e-7
    )



def test_u_validation_loop_emits_exact_physical_diagnostic_names() -> None:
    from src.training.train import _run_epoch

    config = tiny_config("GA_U_physical_state_aux")
    config["training"]["performance"].update({"show_progress_bar": False, "compute_val_metrics": True})
    model = build_model_from_config(config, 86)
    x, terrain, target = batch()
    result = _run_epoch(
        model=model,
        loader=[{"x": x, "y": target, "terrain": terrain}],
        criterion=MultiTaskLoss(config),
        config=config,
        device=torch.device("cpu"),
        input_sequence_length=2,
        input_channels=86,
        output_channels=4,
        train=False,
    )
    for name in ("physical_active_fraction_mae", "physical_canopy_state_mae", "physical_energy_state_mae"):
        assert torch.isfinite(torch.tensor(result[f"val_{name}"]))


@pytest.mark.parametrize("name", [
    "GA_R_mask_guided_regression_attention",
    "GA_S_regression_moe",
    "GA_T_supervised_contrastive",
    "GA_U_physical_state_aux",
    "GAK_resblocks_no_terrain",
])
def test_non_mamba_new_models_have_finite_b4_output_and_loss(name: str) -> None:
    config = tiny_config(name)
    model = build_model_from_config(config, 86).train()
    x, terrain, target = batch()
    output = model(x, terrain=terrain if model.terrain_film_enabled else None)
    assert extract_prediction(output).shape == (4, 4, 4, 4)
    if isinstance(output, dict):
        output["auxiliary_training"] = True
    result = MultiTaskLoss(config)(output, target)
    assert torch.isfinite(result["total_loss"])


@pytest.mark.parametrize(("name", "attention"), [
    ("GPK_mamba_no_terrain", False),
    ("CGPK_temporal_mamba_no_terrain", True),
])
def test_mamba_no_terrain_combinations_have_correct_modules_and_full_output(name: str, attention: bool) -> None:
    if importlib.util.find_spec("mamba_ssm") is None or not torch.cuda.is_available():
        pytest.skip("official mamba_ssm CUDA runtime unavailable")
    config = tiny_config(name)
    model = build_model_from_config(config, 86).cuda().eval()
    assert isinstance(model.post_fusion_backbone, SpatiotemporalMambaBackbone)
    assert isinstance(model.temporal_pooling, TemporalAttentionPooling if attention else TemporalAggregator)
    assert model.decoder is None and model.mask_decoder is not None and model.regression_decoder is not None
    assert model.terrain_film_enabled is False
    with torch.no_grad():
        output = model(torch.randn(4, 2, 86, 4, 4, device="cuda"))
    assert extract_prediction(output).shape == (4, 4, 4, 4)


def test_all_new_configs_registry_protocol_parent_maps_and_old_configs_load() -> None:
    registry = yaml.safe_load(open("configs/ablations/cawfe_latte_ablations.yaml", encoding="utf-8"))["ablations"]
    assert set(NEW_EXPERIMENTS) <= set(registry)
    for name, short_name in NEW_EXPERIMENTS.items():
        entry = registry[name]
        config = load_config(config_path(name))
        assert entry["short_name"] == short_name
        assert entry["parent_architecture"]
        assert entry["changed_components"]
        assert config["model"]["architecture"] == "cawfe_latte"
        assert config["training"]["max_epochs"] == 10
        assert config["training"]["early_stopping"]["enabled"] is False
        assert config["training"]["run_test_after_training"] is False
        assert config["training"]["run_external_test_after_training"] is False
    ga = load_config(config_path("GA_R_mask_guided_regression_attention"))["cawfe_latte"]
    assert ga["post_fusion_backbone"]["type"] == "residual_spatiotemporal" and ga["post_fusion_backbone"]["num_blocks"] == 6
    assert ga["decoder"]["type"] == "separate_regression" and ga["terrain_film"]["enabled"] is True
    gpk = load_config(config_path("GPK_mamba_no_terrain"))["cawfe_latte"]
    assert gpk["post_fusion_backbone"]["type"] == "spatiotemporal_mamba"
    assert gpk["temporal_pooling"]["type"] == "baseline" and gpk["terrain_film"]["enabled"] is False
    cgpk = load_config(config_path("CGPK_temporal_mamba_no_terrain"))["cawfe_latte"]
    assert cgpk["temporal_pooling"]["type"] == "attention" and cgpk["terrain_film"]["enabled"] is False
    gak = build_model_from_config(tiny_config("GAK_resblocks_no_terrain"), 86)
    assert isinstance(gak.post_fusion_backbone, ResidualSpatiotemporalBackbone)
    assert isinstance(gak.temporal_pooling, TemporalAggregator) and gak.terrain_film_enabled is False
    for entry in registry.values():
        load_config(entry["config_path"])
    assert compare_configs() == []


def test_new_parent_delta_no_fire_pareto_and_category_footer() -> None:
    primary = {
        "Val Dice": 0.5, "Val IoU": 0.4, "Val Energy Log MAE": 0.5,
        "Val Surface MAE": 0.5, "Val Canopy MAE": 0.5, "Val Active Canopy MAE": 0.5,
        "No-Fire Pixel FP Rate": 0.2, "No-Fire Patch FP Rate": 0.3,
        "No-Fire Energy Log Pred Mean": 0.1,
    }
    rows = [
        {"Ablation": "baseline", "_short_name": "baseline", "_mtime": 1.0, "_components": [], **primary},
        {"Ablation": "GA_separate_decoder_resblocks", "_short_name": "GA", "_mtime": 1.0, "_components": ["G", "A"], **primary},
        {"Ablation": "GA_R_mask_guided_regression_attention", "_short_name": "GA-R", "_mtime": 2.0, "_components": ["G", "A", "R"], **{**primary, "Val Dice": 0.6, "No-Fire Pixel FP Rate": 0.1}},
    ]
    add_parent_deltas(rows)
    assert rows[-1]["Delta vs GA Val Dice"] == pytest.approx(0.1)
    assert rows[-1]["Delta vs GA No-Fire Pixel FP Rate"] == pytest.approx(-0.1)
    assert "GA-R" in pareto_candidates(rows)
    footer = "\n".join(text_footer(rows))
    assert "Lowest No-Fire Pixel FP" in footer
    assert "MODELS IMPROVING ALL PRIMARY FULL-VALIDATION METRICS VS BASELINE" in footer
    assert "PARETO / NON-DOMINATED CANDIDATES" in footer
