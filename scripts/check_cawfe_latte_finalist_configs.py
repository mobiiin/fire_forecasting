#!/usr/bin/env python3
"""Fail loudly when a CAWFE-Latte finalist config changes protected protocol fields."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from scripts.run_cawfe_latte_finalist import output_parent
from src.config import load_config


DEFAULT_REGISTRY = Path("configs/final_training/cawfe_latte_finalists.yaml")
ABLATION_REGISTRY = Path("configs/ablations/cawfe_latte_ablations.yaml")
EXPECTED_FINALISTS = {
    "baseline": ("baseline", ["baseline"]),
    "CGP": ("CGP_separate_decoder_temporal_attention_mamba", ["C", "G", "P"]),
    "GA_Q2": ("GA_Q2_fire_domain_mmd", ["G", "A", "Q2"]),
    "GPK": ("GPK_mamba_no_terrain", ["G", "P", "K"]),
}
EXPECTED_SEEDS = [42, 123, 2026]
PROVENANCE = {
    "base_config", "config_path", "_config_path", "_config_file_name", "_config_sha256",
    "_base_config_path", "_base_config_sha256",
}


def _common_view(config: Mapping[str, Any]) -> dict[str, Any]:
    view = deepcopy(dict(config))
    for key in PROVENANCE | {"experiment", "final_training", "cawfe_latte"}:
        view.pop(key, None)
    training = dict(view.get("training", {}))
    training.pop("seed", None)
    training.pop("output", None)
    view["training"] = training
    view.pop("seed", None)
    return view


def check_configs(registry_path: Path = DEFAULT_REGISTRY) -> list[str]:
    errors: list[str] = []
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    finalists = payload.get("finalists", {})
    seeds = [int(seed) for seed in payload.get("seeds", [])]
    if set(finalists) != set(EXPECTED_FINALISTS):
        errors.append(f"Finalists must be exactly {list(EXPECTED_FINALISTS)}, got {list(finalists)}.")
    if seeds != EXPECTED_SEEDS:
        errors.append(f"Seeds must be exactly {EXPECTED_SEEDS}, got {seeds}.")

    ablations = (yaml.safe_load(ABLATION_REGISTRY.read_text(encoding="utf-8")) or {}).get("ablations", {})
    configs: dict[str, dict[str, Any]] = {}
    output_paths: set[str] = set()
    for name, (source_name, components) in EXPECTED_FINALISTS.items():
        entry = finalists.get(name)
        if not isinstance(entry, Mapping):
            errors.append(f"{name}: missing registry entry.")
            continue
        if entry.get("source_ablation") != source_name:
            errors.append(f"{name}: source_ablation must be {source_name!r}.")
        if list(entry.get("components", [])) != components:
            errors.append(f"{name}: components must be {components!r}.")
        config_path = Path(str(entry.get("config_path", "")))
        source_path = Path(str(ablations.get(source_name, {}).get("config_path", "")))
        if not config_path.is_file() or not source_path.is_file():
            errors.append(f"{name}: missing final or source config ({config_path}, {source_path}).")
            continue
        config = load_config(config_path)
        source = load_config(source_path)
        configs[name] = config
        if config.get("cawfe_latte") != source.get("cawfe_latte"):
            errors.append(f"{name}: cawfe_latte architecture does not exactly match source ablation {source_name}.")
        if config.get("model") != source.get("model"):
            errors.append(f"{name}: model fields do not exactly match source ablation {source_name}.")
        training = config.get("training", {})
        performance = training.get("performance", {})
        early = training.get("early_stopping", {})
        if int(training.get("max_epochs", -1)) != 60 or int(training.get("epochs", -1)) != 60:
            errors.append(f"{name}: max_epochs and epochs must both be 60.")
        if training.get("max_train_batches_per_epoch") not in (None, "", "null", 0):
            errors.append(f"{name}: training.max_train_batches_per_epoch must be disabled.")
        if performance.get("max_train_batches_per_epoch") not in (None, "", "null", 0):
            errors.append(f"{name}: training.performance.max_train_batches_per_epoch must be disabled.")
        if int(training.get("max_train_batches", -1)) != 7500:
            errors.append(f"{name}: training.max_train_batches must be exactly 7500.")
        data_loader = config.get("data_loader", {}) if isinstance(config.get("data_loader"), Mapping) else {}
        train_loader_config = data_loader.get("train", {}) if isinstance(data_loader.get("train"), Mapping) else {}
        effective_batch_size = int(
            train_loader_config.get("batch_size", data_loader.get("batch_size", training.get("batch_size", config.get("batch_size", -1))))
        )
        if effective_batch_size != 8:
            errors.append(f"{name}: effective training batch_size must be exactly 8, got {effective_batch_size}.")
        if int(training.get("gradient_accumulation_steps", -1)) != 1:
            errors.append(f"{name}: gradient_accumulation_steps must be exactly 1.")
        if bool(training.get("auto_hardware_tuning", {}).get("enabled", False)):
            errors.append(f"{name}: automatic hardware batch tuning must remain disabled.")
        if bool(performance.get("auto_batch_size", False)):
            errors.append(f"{name}: automatic batch-size probing must remain disabled.")
        if bool(performance.get("cudnn_benchmark", True)):
            errors.append(f"{name}: cuDNN benchmarking must be disabled so seeded runs retain deterministic cuDNN behavior.")
        expected_early = {
            "enabled": True, "monitor": "val_loss", "mode": "min", "patience": 8,
            "min_delta": 0.001, "start_epoch": 10,
        }
        for key, expected in expected_early.items():
            if early.get(key) != expected:
                errors.append(f"{name}: early_stopping.{key} must be {expected!r}, got {early.get(key)!r}.")
        if bool(training.get("run_test_after_training", False)) or bool(training.get("run_external_test_after_training", False)):
            errors.append(f"{name}: test evaluation must be disabled.")
        validation = training.get("validation", {})
        screening = validation.get("screening", {})
        full = validation.get("full", {})
        if not bool(screening.get("enabled", False)) or screening.get("sampling") != "stratified_fixed":
            errors.append(f"{name}: deterministic stratified screening validation must be enabled.")
        if screening.get("seed") != 12345:
            errors.append(f"{name}: shared screening seed must remain 12345.")
        if not bool(full.get("enabled", False)) or full.get("checkpoint") != "best":
            errors.append(f"{name}: automatic full validation on the best checkpoint must be enabled.")
        if not bool(full.get("save_analysis_data", False)):
            errors.append(f"{name}: full-validation paper-data export must be enabled.")
        if int(config.get("logging", {}).get("step_log_interval", -1)) != 50:
            errors.append(f"{name}: logging.step_log_interval must be 50.")
        if config.get("final_training", {}).get("finalist") != name:
            errors.append(f"{name}: final_training.finalist is inconsistent.")
        for seed in seeds:
            destination = str(output_parent(name, seed))
            if destination in output_paths:
                errors.append(f"Duplicate architecture/seed output path: {destination}.")
            output_paths.add(destination)

    if configs:
        reference_name = next(iter(EXPECTED_FINALISTS))
        reference = _common_view(configs[reference_name])
        for name, config in configs.items():
            if _common_view(config) != reference:
                errors.append(
                    f"{name}: protected dataset/split/channels/normalization/target/patch/optimizer/LR/loss/"
                    "batch-size/scheduler/training/validation fields differ from baseline."
                )

    baseline = configs.get("baseline", {}).get("cawfe_latte", {})
    cgp = configs.get("CGP", {}).get("cawfe_latte", {})
    ga_q2 = configs.get("GA_Q2", {}).get("cawfe_latte", {})
    gpk = configs.get("GPK", {}).get("cawfe_latte", {})
    if baseline and (
        baseline.get("post_fusion_backbone", {}).get("type") != "baseline_cnn"
        or baseline.get("temporal_pooling", {}).get("type") != "baseline"
    ):
        errors.append("baseline: original baseline post-fusion and pooling are not preserved.")
    if cgp and (
        cgp.get("post_fusion_backbone", {}).get("type") != "spatiotemporal_mamba"
        or cgp.get("post_fusion_backbone", {}).get("backend") != "mamba_ssm"
        or cgp.get("temporal_pooling", {}).get("type") != "attention"
        or cgp.get("decoder", {}).get("type") != "separate_regression"
        or not bool(cgp.get("terrain_film", {}).get("enabled", False))
    ):
        errors.append("CGP: exact C/G/P modules with terrain FiLM ON are not resolved.")
    if ga_q2 and (
        ga_q2.get("post_fusion_backbone", {}).get("type") != "residual_spatiotemporal"
        or ga_q2.get("post_fusion_backbone", {}).get("num_blocks") != 6
        or ga_q2.get("decoder", {}).get("type") != "separate_regression"
        or ga_q2.get("fire_mmd") != {"enabled": True, "loss_weight": 0.05, "kernel": "rbf", "bandwidths": [0.5, 1.0, 2.0, 4.0]}
    ):
        errors.append("GA_Q2: exact G/A/Q2 settings are not resolved.")
    if gpk and (
        gpk.get("post_fusion_backbone", {}).get("type") != "spatiotemporal_mamba"
        or gpk.get("post_fusion_backbone", {}).get("backend") != "mamba_ssm"
        or gpk.get("temporal_pooling", {}).get("type") != "baseline"
        or gpk.get("decoder", {}).get("type") != "separate_regression"
        or bool(gpk.get("terrain_film", {}).get("enabled", True))
    ):
        errors.append("GPK: exact G/P/K settings and baseline pooling are not resolved.")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    errors = check_configs(args.registry)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("CAWFE-Latte finalist config safety check: PASS")
    print("Finalists: baseline, CGP, GA_Q2, GPK")
    print("Seeds: 42, 123, 2026")
    print("Full train dataset preserved; epoch-random 7500-batch subsets enabled; validation-only selection preserved.")


if __name__ == "__main__":
    main()
