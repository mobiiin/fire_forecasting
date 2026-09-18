#!/usr/bin/env python3
"""Train one registered CAWFE-Latte screening ablation and package its results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.config import load_config


REGISTRY_PATH = Path("configs/ablations/cawfe_latte_ablations.yaml")
REQUIRED_METRICS = (
    "loss",
    "mask_dice",
    "mask_iou",
    "mask_precision",
    "mask_recall",
    "energy_log_mae",
    "energy_log_rmse",
    "surface_consumed_mae",
    "surface_consumed_rmse",
    "canopy_consumed_mae",
    "canopy_consumed_rmse",
    "active_surface_consumed_mae",
    "active_surface_consumed_rmse",
    "active_canopy_consumed_mae",
    "active_canopy_consumed_rmse",
    "active_energy_log_mae",
    "active_energy_log_rmse",
)
OPTIONAL_METRICS = (
    "total_patch_count",
    "no_fire_patch_count",
    "fire_patch_count",
    "no_fire_mask_prob_mean",
    "no_fire_mask_false_positive_rate",
    "no_fire_patch_false_positive_rate",
    "no_fire_surface_pred_mean",
    "no_fire_canopy_pred_mean",
    "no_fire_energy_log_pred_mean",
    "no_fire_energy_mw_pred_mean",
)
PATCH_FIRE_METRICS = (
    "patch_fire_accuracy",
    "patch_fire_precision",
    "patch_fire_recall",
    "patch_fire_f1",
)
AUXILIARY_METRICS = (
    "domain_loss",
    "domain_accuracy",
    "domain_random_chance",
    "mmd_loss",
    "mmd_valid_batch_fraction",
    "mask_guidance_alpha",
    "guidance_mean",
    "guidance_std",
    "guidance_mean_active",
    "guidance_mean_inactive",
    "expert_1_mean_weight",
    "expert_2_mean_weight",
    "expert_3_mean_weight",
    "router_entropy",
    "router_max_probability_mean",
    "load_balance_loss",
    *(f"router_{activity}_expert_{expert}_mean_weight" for activity in ("no_fire", "tiny_fire", "small_fire", "medium_fire", "large_fire") for expert in (1, 2, 3)),
    "contrastive_loss",
    "valid_contrastive_anchor_fraction",
    "batch_fraction_no_fire",
    "batch_fraction_tiny",
    "batch_fraction_small",
    "batch_fraction_medium",
    "batch_fraction_large",
    "same_class_cosine_similarity",
    "different_class_cosine_similarity",
    "physical_state_aux_loss",
    "physical_active_fraction_loss",
    "physical_canopy_state_loss",
    "physical_energy_state_loss",
    "physical_active_fraction_mae",
    "physical_canopy_state_mae",
    "physical_energy_state_mae",
)


def load_entry(name: str, registry_path: Path = REGISTRY_PATH) -> dict[str, Any]:
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    entries = payload.get("ablations", {})
    if name not in entries:
        valid = ", ".join(entries)
        raise ValueError(f"Unknown ablation {name!r}. Expected one of: {valid}.")
    return dict(entries[name])


def metric_block(row: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in REQUIRED_METRICS + OPTIONAL_METRICS + PATCH_FIRE_METRICS + AUXILIARY_METRICS:
        key = f"{prefix}_{metric}" if metric != "loss" else f"{prefix}_loss"
        value = row.get(key)
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        result[metric] = value
    return result


def write_history(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ablation")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    entry = load_entry(args.ablation, args.registry)
    config_path = Path(str(entry["config_path"]))
    if args.print_config:
        print(config_path)
        return

    from src.models.model_factory import build_model_from_config
    from src.training.train import train_model_from_config

    config = load_config(config_path)
    if config.get("model", {}).get("architecture") != "cawfe_latte":
        raise ValueError("Ablation training requires model.architecture=cawfe_latte.")
    training = dict(config.get("training", {}))
    if int(training.get("max_epochs", -1)) != 10:
        raise ValueError("Ablation screening requires training.max_epochs=10.")
    if bool(training.get("early_stopping", {}).get("enabled", True)):
        raise ValueError("Ablation screening requires early stopping to be disabled.")
    if bool(training.get("run_test_after_training", False)) or bool(training.get("run_external_test_after_training", False)):
        raise ValueError("Ablation screening must not evaluate the test set.")
    if args.run_id:
        training["run_name"] = args.run_id
        config["training"] = training

    changed_components = list(entry.get("changed_components", []))
    if not changed_components and entry.get("changed_component") not in (None, "none"):
        changed_components = [str(entry["changed_component"])]
    components = [str(value) for value in entry.get("components", [])]
    changed_component_text = ", ".join(changed_components) if changed_components else "none"

    input_channels = int(config.get("model", {}).get("input_channels", 129))
    model = build_model_from_config(config, input_channels=input_channels)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    architecture_summary = (
        f"Ablation: {args.ablation}\n"
        f"Components: {', '.join(components) if components else 'baseline/individual'}\n"
        f"Description: {entry['description']}\n"
        f"Changed components: {changed_component_text}\n"
        f"Model architecture: cawfe_latte\n"
        f"Trainable parameters before lazy spatial-position initialization: {parameter_count}\n\n"
        f"{model}\n"
    )
    del model

    result = train_model_from_config(config)
    run_dir = Path(str(result["run_dir"]))
    rows = [dict(row) for row in result.get("history_rows", [])]
    best_epoch = int(result["best_epoch"])
    best_rows = [row for row in rows if int(row.get("epoch", -1)) == best_epoch]
    if not best_rows:
        raise RuntimeError(f"Best epoch {best_epoch} is missing from training history.")
    best_row = best_rows[-1]
    train_metrics = metric_block(best_row, "train")
    val_metrics = metric_block(best_row, "val")
    final_train_metrics = metric_block(rows[-1], "train")
    final_val_metrics = metric_block(rows[-1], "val")
    epoch_times = [float(row["epoch_time_sec"]) for row in rows if row.get("epoch_time_sec") is not None and math.isfinite(float(row["epoch_time_sec"]))]
    memory_values = [
        float(row[key])
        for row in rows
        for key in ("gpu_memory_allocated_gb", "gpu_memory_reserved_gb")
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    train_time_per_epoch_sec = sum(epoch_times) / len(epoch_times) if epoch_times else None
    peak_gpu_memory_gb = max(memory_values) if memory_values else None

    resolved_source = Path(str(result.get("run_artifact_paths", {}).get("resolved_config_path", "")))
    if not resolved_source.is_file():
        resolved_source = run_dir / "configs" / "resolved_config.yaml"
    shutil.copyfile(resolved_source, run_dir / "resolved_config.yaml")
    write_history(run_dir / "training_history.csv", rows)
    (run_dir / "training_history.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (run_dir / "architecture_summary.txt").write_text(architecture_summary, encoding="utf-8")

    latest_checkpoint = Path(str(result["latest_checkpoint_path"]))
    best_checkpoint = Path(str(result["best_checkpoint_path"]))
    if not latest_checkpoint.is_file() or not best_checkpoint.is_file():
        raise RuntimeError("Both best and final epoch checkpoints are required.")
    shutil.copyfile(latest_checkpoint, run_dir / "checkpoints" / "final_model.pt")

    full_result = result.get("full_validation", {})
    if not isinstance(full_result, Mapping) or not isinstance(full_result.get("metrics"), Mapping):
        raise RuntimeError("Automatic full validation did not return metrics; refusing to mark the ablation complete.")
    full_metrics = dict(full_result["metrics"])
    full_per_fire = dict(full_result.get("per_fire", {}))
    for required in ("full_val_total_patch_count", "full_val_fire_patch_count", "full_val_no_fire_patch_count", "full_val_dice", "full_val_iou"):
        if full_metrics.get(required) is None:
            raise RuntimeError(f"Automatic full validation is missing required metric {required!r}.")
    if not (run_dir / "full_validation_metrics.json").is_file() or not (run_dir / "full_validation_per_fire.json").is_file():
        raise RuntimeError("Automatic full-validation artifacts were not saved inside the run directory.")

    screening_payload = {
        "schema_version": 1,
        "best_epoch": best_epoch,
        "checkpoint": str(best_checkpoint),
        "validation_protocol": result.get("validation", {}),
        "metrics": val_metrics,
    }
    (run_dir / "screening_validation_metrics.json").write_text(
        json.dumps(screening_payload, indent=2) + "\n", encoding="utf-8"
    )

    exact_change = "None; preserved baseline." if args.ablation == "baseline" else entry["description"]
    metrics_payload = {
        "ablation": args.ablation,
        "short_name": entry.get("short_name", args.ablation),
        "components": components,
        "parent_architecture": entry.get("parent_architecture"),
        "description": entry["description"],
        "changed_component": changed_component_text,
        "changed_components": changed_components,
        "exact_change": exact_change,
        "parameter_count": parameter_count,
        "train_time_per_epoch_sec": train_time_per_epoch_sec,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "epochs_trained": int(result.get("epochs_completed", len(rows))),
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_checkpoint),
        "final_checkpoint": str(run_dir / "checkpoints" / "final_model.pt"),
        "training": {
            "epochs_trained": int(result.get("epochs_completed", len(rows))),
            "best_epoch": best_epoch,
            "best_checkpoint": str(best_checkpoint),
            "final_checkpoint": str(run_dir / "checkpoints" / "final_model.pt"),
            "best_epoch_metrics": train_metrics,
            "final_epoch_metrics": final_train_metrics,
        },
        "best_screening_validation": val_metrics,
        "full_validation": full_metrics,
        "full_validation_per_fire": full_per_fire,
        "best_epoch_metrics": {
            "train": train_metrics,
            "validation": val_metrics,
        },
        "final_epoch_metrics": {
            "train": final_train_metrics,
            "validation": final_val_metrics,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    separator = "=" * 60
    lines = [
        f"Ablation name: {args.ablation}",
        f"Description: {entry['description']}",
        f"Exact change from baseline: {exact_change}",
        f"Parameter count: {parameter_count}",
        f"Train time per epoch (seconds): {format_value(train_time_per_epoch_sec)}",
        f"Peak GPU memory (GiB): {format_value(peak_gpu_memory_gb)}",
        f"Epochs trained: {metrics_payload['epochs_trained']}",
        f"Best epoch: {best_epoch}",
        "",
        separator,
        "SCREENING VALIDATION - BEST EPOCH",
        separator,
        f"Best epoch: {best_epoch}",
    ]
    lines.extend(f"  {metric}: {format_value(val_metrics[metric])}" for metric in REQUIRED_METRICS)
    available_optional = [metric for metric in OPTIONAL_METRICS if val_metrics.get(metric) is not None]
    if available_optional:
        lines.extend(["", "Best-epoch validation no-fire metrics:"])
        lines.extend(f"  {metric}: {format_value(val_metrics[metric])}" for metric in available_optional)
    available_patch_fire = [metric for metric in PATCH_FIRE_METRICS if val_metrics.get(metric) is not None]
    if available_patch_fire:
        lines.extend(["", "Best-epoch patch-fire metrics:"])
        lines.extend(f"  train_{metric}: {format_value(train_metrics[metric])}" for metric in available_patch_fire)
        lines.extend(f"  val_{metric}: {format_value(val_metrics[metric])}" for metric in available_patch_fire)
    available_train_auxiliary = [metric for metric in AUXILIARY_METRICS if train_metrics.get(metric) is not None]
    if available_train_auxiliary:
        lines.extend(["", "Best-epoch architecture-specific training diagnostics:"])
        lines.extend(f"  {metric}: {format_value(train_metrics[metric])}" for metric in available_train_auxiliary)
    available_val_auxiliary = [metric for metric in AUXILIARY_METRICS if val_metrics.get(metric) is not None]
    if available_val_auxiliary:
        lines.extend(["", "Best-epoch architecture-specific validation diagnostics:"])
        lines.extend(f"  {metric}: {format_value(val_metrics[metric])}" for metric in available_val_auxiliary)
    if final_train_metrics.get("mask_guidance_alpha") is not None:
        lines.extend(["", f"Final learned mask_guidance_alpha: {format_value(final_train_metrics['mask_guidance_alpha'])}"])
    lines.extend(
        [
            "",
            separator,
            "FULL VALIDATION - BEST CHECKPOINT",
            separator,
            f"Total patches: {format_value(full_metrics.get('full_val_total_patch_count'))}",
            f"Fire patches: {format_value(full_metrics.get('full_val_fire_patch_count'))}",
            f"No-fire patches: {format_value(full_metrics.get('full_val_no_fire_patch_count'))}",
            f"Fire %: {format_value(full_metrics.get('full_val_fire_patch_percent'))}",
            f"No-fire %: {format_value(full_metrics.get('full_val_no_fire_patch_percent'))}",
            "",
            f"Dice: {format_value(full_metrics.get('full_val_dice'))}",
            f"IoU: {format_value(full_metrics.get('full_val_iou'))}",
            f"Energy Log MAE: {format_value(full_metrics.get('full_val_energy_log_mae'))}",
            f"Surface MAE: {format_value(full_metrics.get('full_val_surface_mae'))}",
            f"Canopy MAE: {format_value(full_metrics.get('full_val_canopy_mae'))}",
            f"Active Canopy MAE: {format_value(full_metrics.get('full_val_active_canopy_mae'))}",
            "",
            f"No-Fire Mask Probability Mean: {format_value(full_metrics.get('full_val_no_fire_mask_prob_mean'))}",
            f"No-Fire Pixel FP Rate: {format_value(full_metrics.get('full_val_no_fire_mask_false_positive_rate'))}",
            f"No-Fire Patch FP Rate: {format_value(full_metrics.get('full_val_no_fire_patch_false_positive_rate'))}",
            f"No-Fire Surface Pred Mean: {format_value(full_metrics.get('full_val_no_fire_surface_pred_mean'))}",
            f"No-Fire Canopy Pred Mean: {format_value(full_metrics.get('full_val_no_fire_canopy_pred_mean'))}",
            f"No-Fire Energy Log Pred Mean: {format_value(full_metrics.get('full_val_no_fire_energy_log_pred_mean'))}",
        ]
    )
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
