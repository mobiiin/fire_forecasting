#!/usr/bin/env python3
"""Fail when a CAWFE-Latte screening config changes anything unintended."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from src.config import load_config


DEFAULT_REGISTRY = Path("configs/ablations/cawfe_latte_ablations.yaml")
PROVENANCE_KEYS = {
    "base_config",
    "config_path",
    "_config_path",
    "_config_file_name",
    "_config_sha256",
    "_base_config_path",
    "_base_config_sha256",
}
COMMON_ALLOWED = {
    "experiment.name",
    "cawfe_latte.ablation.name",
    "training.output.root_dir",
}
ARCHITECTURE_ALLOWED = {
    "baseline": set(),
    "A_resblocks": {
        "cawfe_latte.post_fusion_backbone.type",
        "cawfe_latte.post_fusion_backbone.num_blocks",
    },
    "B_multiscale_context": {
        "cawfe_latte.post_fusion_backbone.type",
    },
    "C_temporal_attention": {
        "cawfe_latte.temporal_pooling.type",
    },
    "G_separate_regression_decoder": {
        "cawfe_latte.decoder.type",
    },
    "I_patch_fire_classifier": {
        "cawfe_latte.patch_fire_head.enabled",
        "cawfe_latte.patch_fire_head.hidden_dim",
        "cawfe_latte.patch_fire_head.dropout",
        "training.loss.patch_fire.enabled",
        "training.loss.patch_fire.weight",
    },
    "K_no_terrain_film": {
        "cawfe_latte.terrain_film.enabled",
    },
    "L_simple_concat_fusion": {
        "cawfe_latte.fusion.type",
    },
    "D_local_window_attention": {
        "cawfe_latte.post_fusion_backbone.type",
        "cawfe_latte.post_fusion_backbone.num_blocks",
        "cawfe_latte.post_fusion_backbone.dim",
        "cawfe_latte.post_fusion_backbone.num_heads",
        "cawfe_latte.post_fusion_backbone.window_size",
        "cawfe_latte.post_fusion_backbone.mlp_ratio",
        "cawfe_latte.post_fusion_backbone.dropout",
    },
    "E_earthformer_lite": {
        "cawfe_latte.post_fusion_backbone.type",
        "cawfe_latte.post_fusion_backbone.dim",
        "cawfe_latte.post_fusion_backbone.num_blocks",
        "cawfe_latte.post_fusion_backbone.num_heads",
        "cawfe_latte.post_fusion_backbone.cuboid_size[0]",
        "cawfe_latte.post_fusion_backbone.cuboid_size[1]",
        "cawfe_latte.post_fusion_backbone.cuboid_size[2]",
        "cawfe_latte.post_fusion_backbone.mlp_ratio",
        "cawfe_latte.post_fusion_backbone.dropout",
        "cawfe_latte.post_fusion_backbone.use_shifted_cuboids",
    },
    "O_fourier_postfusion": {
        "cawfe_latte.post_fusion_backbone.type",
        "cawfe_latte.post_fusion_backbone.dim",
        "cawfe_latte.post_fusion_backbone.num_blocks",
        "cawfe_latte.post_fusion_backbone.modes_h",
        "cawfe_latte.post_fusion_backbone.modes_w",
        "cawfe_latte.post_fusion_backbone.residual",
        "cawfe_latte.post_fusion_backbone.dropout",
    },
    "P_mamba_postfusion": {
        "cawfe_latte.post_fusion_backbone.type",
        "cawfe_latte.post_fusion_backbone.dim",
        "cawfe_latte.post_fusion_backbone.num_blocks",
        "cawfe_latte.post_fusion_backbone.expansion",
        "cawfe_latte.post_fusion_backbone.dropout",
        "cawfe_latte.post_fusion_backbone.scan_order",
        "cawfe_latte.post_fusion_backbone.backend",
    },
}
EXPECTED_COMPONENT = {
    "baseline": "none",
    "A_resblocks": "post_fusion_backbone",
    "B_multiscale_context": "post_fusion_backbone",
    "C_temporal_attention": "temporal_pooling",
    "G_separate_regression_decoder": "decoder",
    "I_patch_fire_classifier": "patch_fire_head",
    "K_no_terrain_film": "terrain_film",
    "L_simple_concat_fusion": "fusion",
    "D_local_window_attention": "post_fusion_backbone",
    "E_earthformer_lite": "post_fusion_backbone",
    "O_fourier_postfusion": "post_fusion_backbone",
    "P_mamba_postfusion": "post_fusion_backbone",
}
EXPECTED_COMPUTE_CLASS = {
    "D_local_window_attention": "medium",
    "E_earthformer_lite": "high",
    "O_fourier_postfusion": "medium",
    "P_mamba_postfusion": "high",
}
COMBINATION_METADATA = {
    "CG_separate_decoder_temporal_attention": {
        "short_name": "CG", "components": ["C", "G"],
        "changed_components": ["temporal_pooling", "decoder"], "compute": "medium",
    },
    "GA_separate_decoder_resblocks": {
        "short_name": "GA", "components": ["G", "A"],
        "changed_components": ["post_fusion_backbone", "decoder"], "compute": "medium",
    },
    "GE_separate_decoder_earthformer": {
        "short_name": "GE", "components": ["G", "E"],
        "changed_components": ["post_fusion_backbone", "decoder"], "compute": "high",
    },
    "GP_separate_decoder_mamba": {
        "short_name": "GP", "components": ["G", "P"],
        "changed_components": ["post_fusion_backbone", "decoder"], "compute": "high",
    },
    "GK_separate_decoder_no_terrain": {
        "short_name": "GK", "components": ["G", "K"],
        "changed_components": ["decoder", "terrain_film"], "compute": "medium",
    },
    "CGA_separate_decoder_temporal_attention_resblocks": {
        "short_name": "CGA", "components": ["C", "G", "A"],
        "changed_components": ["post_fusion_backbone", "temporal_pooling", "decoder"], "compute": "high",
    },
    "CGE_separate_decoder_temporal_attention_earthformer": {
        "short_name": "CGE", "components": ["C", "G", "E"],
        "changed_components": ["post_fusion_backbone", "temporal_pooling", "decoder"], "compute": "high",
    },
    "CGP_separate_decoder_temporal_attention_mamba": {
        "short_name": "CGP", "components": ["C", "G", "P"],
        "changed_components": ["post_fusion_backbone", "temporal_pooling", "decoder"], "compute": "high",
    },
    "CGK_separate_decoder_temporal_attention_no_terrain": {
        "short_name": "CGK", "components": ["C", "G", "K"],
        "changed_components": ["temporal_pooling", "decoder", "terrain_film"], "compute": "medium",
    },
}
COMPONENT_CONFIG_NAMES = {
    "A": "A_resblocks", "C": "C_temporal_attention", "E": "E_earthformer_lite",
    "G": "G_separate_regression_decoder", "K": "K_no_terrain_film", "P": "P_mamba_postfusion",
}
for combination_name, metadata in COMBINATION_METADATA.items():
    ARCHITECTURE_ALLOWED[combination_name] = set().union(
        *(ARCHITECTURE_ALLOWED[COMPONENT_CONFIG_NAMES[code]] for code in metadata["components"])
    )
    EXPECTED_COMPUTE_CLASS[combination_name] = metadata["compute"]


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings/lists into comparable dotted leaves."""
    if isinstance(value, Mapping):
        leaves: dict[str, Any] = {}
        for key, nested in value.items():
            if not prefix and str(key) in PROVENANCE_KEYS:
                continue
            dotted = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(flatten(nested, dotted))
        return leaves
    if isinstance(value, list):
        return {f"{prefix}[{index}]": nested for index, nested in enumerate(value)}
    return {prefix: value}


def compare_configs(registry_path: Path = DEFAULT_REGISTRY) -> list[str]:
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    entries = payload.get("ablations", {})
    expected_names = set(ARCHITECTURE_ALLOWED)
    errors: list[str] = []
    if set(entries) != expected_names:
        errors.append(f"Registry entries must be exactly {sorted(expected_names)}, got {sorted(entries)}.")
        return errors

    configs: dict[str, dict[str, Any]] = {}
    for name, entry in entries.items():
        if entry.get("name") != name:
            errors.append(f"{name}: registry name field is {entry.get('name')!r}.")
        if name in EXPECTED_COMPONENT and entry.get("changed_component") != EXPECTED_COMPONENT[name]:
            errors.append(
                f"{name}: changed_component must be {EXPECTED_COMPONENT[name]!r}, "
                f"got {entry.get('changed_component')!r}."
            )
        if name in COMBINATION_METADATA:
            metadata = COMBINATION_METADATA[name]
            for field in ("short_name", "components", "changed_components"):
                if entry.get(field) != metadata[field]:
                    errors.append(f"{name}: {field} must be {metadata[field]!r}, got {entry.get(field)!r}.")
        if name in EXPECTED_COMPUTE_CLASS and entry.get("expected_compute_class") != EXPECTED_COMPUTE_CLASS[name]:
            errors.append(
                f"{name}: expected_compute_class must be {EXPECTED_COMPUTE_CLASS[name]!r}, "
                f"got {entry.get('expected_compute_class')!r}."
            )
        path = Path(str(entry.get("config_path", "")))
        if not path.is_file():
            errors.append(f"{name}: config does not exist: {path}.")
            continue
        config = load_config(path)
        configs[name] = config
        if config.get("model", {}).get("architecture") != "cawfe_latte":
            errors.append(f"{name}: model.architecture must be cawfe_latte.")
        if int(config.get("training", {}).get("max_epochs", -1)) != 10:
            errors.append(f"{name}: training.max_epochs must be 10.")
        if bool(config.get("training", {}).get("early_stopping", {}).get("enabled", True)):
            errors.append(f"{name}: training.early_stopping.enabled must be false.")
        if bool(config.get("training", {}).get("run_test_after_training", True)):
            errors.append(f"{name}: training.run_test_after_training must be false.")
        if bool(config.get("training", {}).get("run_external_test_after_training", True)):
            errors.append(f"{name}: training.run_external_test_after_training must be false.")

    if "baseline" not in configs:
        return errors
    baseline = flatten(configs["baseline"])
    for name in (entry_name for entry_name in ARCHITECTURE_ALLOWED if entry_name != "baseline"):
        if name not in configs:
            continue
        candidate = flatten(configs[name])
        differences = {
            key
            for key in set(baseline) | set(candidate)
            if baseline.get(key, object()) != candidate.get(key, object())
        }
        allowed = COMMON_ALLOWED | ARCHITECTURE_ALLOWED[name]
        unexpected = sorted(differences - allowed)
        missing_architecture_change = sorted(ARCHITECTURE_ALLOWED[name] - differences)
        if unexpected:
            errors.append(f"{name}: unexpected differences: {', '.join(unexpected)}.")
        if missing_architecture_change:
            errors.append(f"{name}: missing expected differences: {', '.join(missing_architecture_change)}.")
        if name in COMBINATION_METADATA:
            for component in COMBINATION_METADATA[name]["components"]:
                parent_name = COMPONENT_CONFIG_NAMES[component]
                parent = flatten(configs[parent_name])
                for key in ARCHITECTURE_ALLOWED[parent_name]:
                    if candidate.get(key, object()) != parent.get(key, object()):
                        errors.append(f"{name}: {key} must exactly match isolated parent {parent_name}.")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    errors = compare_configs(args.registry)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("CAWFE-Latte ablation config diff check: PASS")
    print("Protected data, split, normalization, channels, encoder, unrelated model, optimizer, LR, batch-size, target, and seed settings are identical.")


if __name__ == "__main__":
    main()
