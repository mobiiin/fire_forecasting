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
    "energy_log_mae",
    "surface_consumed_mae",
    "canopy_consumed_mae",
    "active_canopy_consumed_mae",
    "active_energy_log_mae",
)
OPTIONAL_METRICS = (
    "no_fire_patch_count",
    "no_fire_mask_prob_mean",
    "no_fire_mask_false_positive_rate",
    "no_fire_surface_pred_mean",
    "no_fire_canopy_pred_mean",
    "no_fire_energy_log_pred_mean",
)
PATCH_FIRE_METRICS = (
    "patch_fire_accuracy",
    "patch_fire_precision",
    "patch_fire_recall",
    "patch_fire_f1",
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
    for metric in REQUIRED_METRICS + OPTIONAL_METRICS + PATCH_FIRE_METRICS:
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
    (run_dir / "architecture_summary.txt").write_text(architecture_summary, encoding="utf-8")

    latest_checkpoint = Path(str(result["latest_checkpoint_path"]))
    best_checkpoint = Path(str(result["best_checkpoint_path"]))
    if not latest_checkpoint.is_file() or not best_checkpoint.is_file():
        raise RuntimeError("Both best and final epoch checkpoints are required.")
    shutil.copyfile(latest_checkpoint, run_dir / "checkpoints" / "final_model.pt")

    exact_change = "None; preserved baseline." if args.ablation == "baseline" else entry["description"]
    metrics_payload = {
        "ablation": args.ablation,
        "short_name": entry.get("short_name", args.ablation),
        "components": components,
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
        "best_epoch_metrics": {
            "train": train_metrics,
            "validation": val_metrics,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

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
        "Best-epoch train metrics:",
    ]
    lines.extend(f"  {metric}: {format_value(train_metrics[metric])}" for metric in REQUIRED_METRICS)
    lines.extend(["", "Best-epoch validation metrics:"])
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
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
