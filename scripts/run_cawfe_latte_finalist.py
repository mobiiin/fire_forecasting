#!/usr/bin/env python3
"""Train one registered CAWFE-Latte finalist seed and package final-run artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
from typing import Any, Mapping

import yaml

from src.config import load_config


REGISTRY_PATH = Path("configs/final_training/cawfe_latte_finalists.yaml")
OUTPUT_ROOT = Path("artifacts/final_training/cawfe_latte")

HISTORY_ALIASES = {
    "train_total_loss": "train_loss",
    "screening_val_total_loss": "val_loss",
    "train_surface_loss": "train_loss_surface",
    "screening_val_surface_loss": "val_loss_surface",
    "train_canopy_loss": "train_loss_canopy",
    "screening_val_canopy_loss": "val_loss_canopy",
    "train_mask_loss": "train_loss_mask_total",
    "screening_val_mask_loss": "val_loss_mask_total",
    "train_energy_loss": "train_loss_energy",
    "screening_val_energy_loss": "val_loss_energy",
    "train_dice": "train_mask_dice",
    "screening_val_dice": "val_mask_dice",
    "train_iou": "train_mask_iou",
    "screening_val_iou": "val_mask_iou",
    "train_surface_mae": "train_surface_consumed_mae",
    "screening_val_surface_mae": "val_surface_consumed_mae",
    "train_canopy_mae": "train_canopy_consumed_mae",
    "screening_val_canopy_mae": "val_canopy_consumed_mae",
    "train_energy_log_mae": "train_energy_log_mae",
    "screening_val_energy_log_mae": "val_energy_log_mae",
    "train_active_canopy_mae": "train_active_canopy_consumed_mae",
    "screening_val_active_canopy_mae": "val_active_canopy_consumed_mae",
    "epoch_time_seconds": "epoch_time_sec",
    "mmd_valid_batch_fraction": "train_mmd_valid_batch_fraction",
}


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload.get("finalists"), Mapping) or not isinstance(payload.get("seeds"), list):
        raise ValueError(f"Invalid finalist registry: {path}")
    return payload


def load_entry(name: str, registry_path: Path = REGISTRY_PATH) -> tuple[dict[str, Any], list[int]]:
    payload = load_registry(registry_path)
    finalists = payload["finalists"]
    if name not in finalists:
        raise ValueError(f"Unknown finalist {name!r}. Expected one of: {', '.join(finalists)}.")
    return dict(finalists[name]), [int(seed) for seed in payload["seeds"]]


def output_parent(finalist: str, seed: int, root: Path = OUTPUT_ROOT) -> Path:
    return root / finalist / f"seed_{int(seed)}"


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _finite_values(rows: list[Mapping[str, Any]], keys: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
    return values


def history_with_aliases(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        aliased = {name: row.get(original) for name, original in HISTORY_ALIASES.items()}
        aliased.update(row)
        converted.append(aliased)
    return converted


PAPER_HISTORY_FIELDS = {
    "train_total_loss": "train_loss",
    "train_surface_loss": "train_loss_surface",
    "train_canopy_loss": "train_loss_canopy",
    "train_mask_loss": "train_loss_mask_total",
    "train_energy_loss": "train_loss_energy",
    "train_dice": "train_mask_dice",
    "train_iou": "train_mask_iou",
    "train_surface_mae": "train_surface_consumed_mae",
    "train_surface_rmse": "train_surface_consumed_rmse",
    "train_canopy_mae": "train_canopy_consumed_mae",
    "train_canopy_rmse": "train_canopy_consumed_rmse",
    "train_energy_log_mae": "train_energy_log_mae",
    "train_energy_log_rmse": "train_energy_log_rmse",
    "train_active_canopy_mae": "train_active_canopy_consumed_mae",
    "train_active_energy_log_mae": "train_active_energy_log_mae",
    "val_total_loss": "val_loss",
    "val_surface_loss": "val_loss_surface",
    "val_canopy_loss": "val_loss_canopy",
    "val_mask_loss": "val_loss_mask_total",
    "val_energy_loss": "val_loss_energy",
    "val_dice": "val_mask_dice",
    "val_iou": "val_mask_iou",
    "val_surface_mae": "val_surface_consumed_mae",
    "val_surface_rmse": "val_surface_consumed_rmse",
    "val_canopy_mae": "val_canopy_consumed_mae",
    "val_canopy_rmse": "val_canopy_consumed_rmse",
    "val_energy_log_mae": "val_energy_log_mae",
    "val_energy_log_rmse": "val_energy_log_rmse",
    "val_active_canopy_mae": "val_active_canopy_consumed_mae",
    "val_active_energy_log_mae": "val_active_energy_log_mae",
    "val_no_fire_mask_prob_mean": "val_no_fire_mask_prob_mean",
    "val_no_fire_pixel_fp_rate": "val_no_fire_mask_false_positive_rate",
    "val_no_fire_patch_fp_rate": "val_no_fire_patch_false_positive_rate",
    "val_no_fire_surface_abs_mean": "val_no_fire_surface_abs_pred_mean",
    "val_no_fire_canopy_abs_mean": "val_no_fire_canopy_abs_pred_mean",
    "val_no_fire_energy_log_abs_mean": "val_no_fire_energy_log_abs_pred_mean",
    "learning_rate": "learning_rate",
    "gradient_norm": "train_gradient_norm",
    "epoch_train_seconds": "train_epoch_seconds",
    "epoch_val_seconds": "val_epoch_seconds",
    "epoch_total_seconds": "epoch_time_sec",
    "peak_gpu_memory_gb": "gpu_memory_allocated_gb",
    "train_mmd_loss": "train_mmd_loss",
    "mmd_valid_batch_fraction": "train_mmd_valid_batch_fraction",
}


def paper_epoch_history(
    rows: list[Mapping[str, Any]],
    *,
    model_name: str,
    architecture_name: str,
    seed: int,
    best_epoch: int,
) -> list[dict[str, Any]]:
    """Build a stable, publication-oriented epoch schema without discarding raw fields."""

    converted: list[dict[str, Any]] = []
    for source in rows:
        epoch = int(source["epoch"])
        row: dict[str, Any] = {
            "model_name": model_name,
            "architecture_name": architecture_name,
            "seed": int(seed),
            "epoch": epoch,
        }
        row.update({
            name: (None if isinstance(source.get(raw_name), float) and not math.isfinite(source[raw_name]) else source.get(raw_name))
            for name, raw_name in PAPER_HISTORY_FIELDS.items()
        })
        row["is_best_epoch"] = int(epoch == int(best_epoch))
        selection_value = source.get("val_loss")
        row["checkpoint_selection_metric"] = (
            None if isinstance(selection_value, float) and not math.isfinite(selection_value) else selection_value
        )
        row["checkpoint_selection_metric_name"] = "val_loss"
        # Preserve every raw/auxiliary value as an additional column. This keeps
        # future analyses possible when a method reports architecture-specific diagnostics.
        for key, value in source.items():
            if key not in row:
                row[str(key)] = None if isinstance(value, float) and not math.isfinite(value) else value
        converted.append(row)
    epochs = [int(row["epoch"]) for row in converted]
    if len(epochs) != len(set(epochs)):
        raise RuntimeError("Epoch history contains duplicate epoch rows.")
    if sum(int(row["is_best_epoch"]) for row in converted) != 1:
        raise RuntimeError(f"Epoch history does not mark exactly one best epoch: {best_epoch}")
    return converted


def write_history(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _best_validation_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = {
        str(key).removeprefix("val_"): value
        for key, value in row.items()
        if str(key).startswith("val_")
    }
    metrics["total_loss"] = row.get("val_loss")
    return metrics


def _validate_config(config: Mapping[str, Any], finalist: str, seed: int) -> None:
    training = config.get("training", {})
    performance = training.get("performance", {})
    validation = training.get("validation", {})
    early = training.get("early_stopping", {})
    problems: list[str] = []
    if config.get("model", {}).get("architecture") != "cawfe_latte":
        problems.append("model.architecture must be cawfe_latte")
    if int(training.get("max_epochs", -1)) != 60 or int(training.get("epochs", -1)) != 60:
        problems.append("training.max_epochs and training.epochs must both be 60")
    if performance.get("max_train_batches_per_epoch") not in (None, "", "null", 0):
        problems.append("training.performance.max_train_batches_per_epoch must be disabled")
    if training.get("max_train_batches_per_epoch") not in (None, "", "null", 0):
        problems.append("training.max_train_batches_per_epoch must be disabled")
    if int(training.get("max_train_batches", -1)) != 7500:
        problems.append("training.max_train_batches must be exactly 7500")
    data_loader = config.get("data_loader", {}) if isinstance(config.get("data_loader"), Mapping) else {}
    train_loader_config = data_loader.get("train", {}) if isinstance(data_loader.get("train"), Mapping) else {}
    effective_batch_size = int(
        train_loader_config.get("batch_size", data_loader.get("batch_size", training.get("batch_size", config.get("batch_size", -1))))
    )
    if effective_batch_size != 8:
        problems.append(f"effective training batch_size must be exactly 8, got {effective_batch_size}")
    if int(training.get("gradient_accumulation_steps", -1)) != 1:
        problems.append("training.gradient_accumulation_steps must be exactly 1")
    if bool(training.get("auto_hardware_tuning", {}).get("enabled", False)):
        problems.append("training.auto_hardware_tuning must remain disabled")
    if bool(performance.get("auto_batch_size", False)):
        problems.append("training.performance.auto_batch_size must remain disabled")
    expected_early = {"enabled": True, "monitor": "val_loss", "mode": "min", "patience": 8, "min_delta": 0.001, "start_epoch": 10}
    for key, expected in expected_early.items():
        if early.get(key) != expected:
            problems.append(f"training.early_stopping.{key} must be {expected!r}")
    if bool(training.get("run_test_after_training", False)) or bool(training.get("run_external_test_after_training", False)):
        problems.append("held-out test evaluation must be disabled")
    screening = validation.get("screening", {})
    full = validation.get("full", {})
    if not bool(screening.get("enabled", False)) or screening.get("sampling") != "stratified_fixed":
        problems.append("deterministic stratified screening validation must be enabled")
    if not bool(full.get("enabled", False)) or full.get("checkpoint") != "best":
        problems.append("automatic full validation of the best checkpoint must be enabled")
    if not bool(full.get("save_analysis_data", False)):
        problems.append("training.validation.full.save_analysis_data must be enabled")
    if int(config.get("logging", {}).get("step_log_interval", -1)) != 50:
        problems.append("logging.step_log_interval must be 50")
    if int(training.get("seed", -1)) != int(seed):
        problems.append(f"resolved training seed must be {seed}")
    if config.get("final_training", {}).get("finalist") != finalist:
        problems.append(f"final_training.finalist must be {finalist}")
    if problems:
        raise ValueError("Unsafe finalist config:\n- " + "\n- ".join(problems))


def _format(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("finalist", nargs="?")
    parser.add_argument("seed", nargs="?", type=int)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--print-output-parent", action="store_true")
    parser.add_argument("--list-finalists", action="store_true")
    parser.add_argument("--list-seeds", action="store_true")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    if args.list_finalists:
        print("\n".join(registry["finalists"]))
        return
    if args.list_seeds:
        print("\n".join(str(int(seed)) for seed in registry["seeds"]))
        return
    if args.finalist is None:
        parser.error("finalist is required")
    entry, seeds = load_entry(args.finalist, args.registry)
    config_path = Path(str(entry["config_path"]))
    if args.print_config:
        print(config_path)
        return
    if args.seed is None:
        parser.error("seed is required for a training run")
    if int(args.seed) not in seeds:
        raise ValueError(f"Seed {args.seed} is not registered. Expected one of: {seeds}.")
    if args.print_output_parent:
        print(output_parent(args.finalist, args.seed))
        return

    import torch
    from src.models.model_factory import build_model_from_config
    from src.training.checkpoints import load_checkpoint, load_model_state_dict_compatible
    from src.training.train import train_model_from_config

    started_at = datetime.now(timezone.utc)
    config = load_config(config_path)
    training = dict(config.get("training", {}))
    training["seed"] = int(args.seed)
    training["max_epochs"] = 60
    training["epochs"] = 60
    training["max_train_batches_per_epoch"] = None
    training["max_train_batches"] = 7500
    performance = dict(training.get("performance", {}))
    performance["max_train_batches_per_epoch"] = None
    training["performance"] = performance
    output = dict(training.get("output", {}))
    output["root_dir"] = str(output_parent(args.finalist, args.seed))
    output["flat_run_layout"] = True
    output["update_architecture_latest_checkpoint"] = False
    training["output"] = output
    if args.run_id:
        training["run_name"] = args.run_id
        training["overwrite_run"] = True
    config["training"] = training
    config["seed"] = int(args.seed)
    _validate_config(config, args.finalist, args.seed)

    input_channels = int(config.get("model", {}).get("input_channels", 129))
    model = build_model_from_config(config, input_channels=input_channels)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model_description = str(model)
    del model

    result = train_model_from_config(config)
    ended_at = datetime.now(timezone.utc)
    run_dir = Path(str(result["run_dir"]))
    raw_rows = [dict(row) for row in result.get("history_rows", [])]
    rows = history_with_aliases(raw_rows)
    if not rows:
        raise RuntimeError("Training returned no epoch history.")
    best_epoch = int(result["best_epoch"])
    best_rows = [row for row in rows if int(row.get("epoch", -1)) == best_epoch]
    if not best_rows:
        raise RuntimeError(f"Best epoch {best_epoch} is missing from training history.")
    best_row = best_rows[-1]

    resolved_source = Path(str(result.get("run_artifact_paths", {}).get("resolved_config_path", "")))
    if not resolved_source.is_file():
        resolved_source = run_dir / "configs" / "resolved_config.yaml"
    shutil.copyfile(resolved_source, run_dir / "resolved_config.yaml")
    write_history(run_dir / "training_history.csv", rows)
    _json_write(run_dir / "training_history.json", {"rows": rows})
    paper_rows = paper_epoch_history(
        raw_rows,
        model_name=args.finalist,
        architecture_name=str(entry["source_ablation"]),
        seed=int(args.seed),
        best_epoch=best_epoch,
    )
    epoch_history_csv = run_dir / "history" / "epoch_history.csv"
    epoch_history_json = run_dir / "history" / "epoch_history.json"
    write_history(epoch_history_csv, paper_rows)
    _json_write(epoch_history_json, paper_rows)
    if len(paper_rows) != int(result.get("epochs_completed", len(paper_rows))):
        raise RuntimeError("Epoch-history row count does not equal the number of completed epochs.")

    latest_checkpoint = Path(str(result["latest_checkpoint_path"]))
    best_checkpoint = Path(str(result["best_checkpoint_path"]))
    if not latest_checkpoint.is_file() or not best_checkpoint.is_file():
        raise RuntimeError("Both best and latest checkpoints are required.")
    materialized_model = build_model_from_config(config, input_channels=input_channels)
    checkpoint_payload = load_checkpoint(best_checkpoint, map_location="cpu")
    load_model_state_dict_compatible(materialized_model, checkpoint_payload, best_checkpoint)
    total_parameters = sum(parameter.numel() for parameter in materialized_model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in materialized_model.parameters() if parameter.requires_grad)
    model_size_bytes = sum(
        int(tensor.numel()) * int(tensor.element_size())
        for tensor in materialized_model.state_dict().values()
        if torch.is_tensor(tensor)
    )
    del materialized_model, checkpoint_payload
    architecture_summary = (
        f"Finalist: {args.finalist}\n"
        f"Source ablation: {entry['source_ablation']}\n"
        f"Components: {', '.join(str(value) for value in entry.get('components', []))}\n"
        f"Seed: {args.seed}\n"
        f"Model architecture: cawfe_latte\n"
        f"Trainable parameter count: {trainable_parameters}\n"
        f"Total parameter count: {total_parameters}\n\n"
        f"{model_description}\n"
    )
    (run_dir / "architecture_summary.txt").write_text(architecture_summary, encoding="utf-8")
    final_checkpoint = run_dir / "checkpoints" / "final_model.pt"
    shutil.copy2(latest_checkpoint, final_checkpoint)

    full_result = result.get("full_validation", {})
    if not isinstance(full_result, Mapping) or not isinstance(full_result.get("metrics"), Mapping):
        raise RuntimeError("Automatic full validation did not return metrics.")
    full_metrics = dict(full_result["metrics"])
    required_artifacts = (
        "full_validation_metrics.json",
        "full_validation_per_fire.json",
        "full_validation_by_activity_bin.json",
        "evaluation/per_fire_validation_metrics.csv",
        "evaluation/validation_sample_metrics.csv",
        "evaluation/qualitative_predictions.npz",
        "metadata/training_sampling_protocol.json",
    )
    missing = [name for name in required_artifacts if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing automatic full-validation artifacts: {missing}")

    best_validation = {
        "schema_version": 1,
        "metric_scope": "deterministic_screening_validation_best_epoch",
        "best_epoch": best_epoch,
        "checkpoint": str(best_checkpoint),
        "validation_protocol": result.get("validation", {}),
        "metrics": _best_validation_metrics(best_row),
    }
    _json_write(run_dir / "best_validation_metrics.json", best_validation)

    train_epoch_times = _finite_values(raw_rows, ("train_epoch_seconds",))
    epoch_times = _finite_values(raw_rows, ("epoch_time_sec",))
    memory_values = _finite_values(raw_rows, ("gpu_memory_allocated_gb",))
    efficiency = {
        "parameter_count": total_parameters,
        "total_parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "model_size_mb": model_size_bytes / float(1024**2),
        "mean_train_seconds_per_epoch": sum(train_epoch_times) / len(train_epoch_times) if train_epoch_times else None,
        "mean_epoch_time_seconds": sum(epoch_times) / len(epoch_times) if epoch_times else None,
        "total_training_time_seconds": sum(epoch_times) if epoch_times else None,
        "peak_gpu_memory_gb": max(memory_values) if memory_values else None,
        "inference_time_per_batch_ms": full_metrics.get("full_val_inference_time_per_batch_ms"),
        "inference_time_per_sample_ms": full_metrics.get("full_val_inference_time_per_sample_ms"),
        "throughput_samples_per_second": full_metrics.get("full_val_samples_per_second"),
        "inference_time_seconds": full_metrics.get("full_val_inference_time_seconds"),
        "samples_per_second": full_metrics.get("full_val_samples_per_second"),
    }
    efficiency_path = run_dir / "evaluation" / "efficiency_metrics.json"
    _json_write(
        efficiency_path,
        {
            "schema_version": 1,
            "model_name": args.finalist,
            "seed": int(args.seed),
            **efficiency,
        },
    )

    validation_identity = result.get("validation", {}).get("dataset_identity", {})
    normalization = result.get("normalization", {})
    train_normalization = normalization.get("train", {}) if isinstance(normalization, Mapping) else {}
    normalization_path_value = train_normalization.get("stats_path") if isinstance(train_normalization, Mapping) else None
    normalization_path = Path(str(normalization_path_value)).expanduser() if normalization_path_value else None
    metadata = {
        "schema_version": 1,
        "architecture": args.finalist,
        "model_architecture": "cawfe_latte",
        "source_ablation": entry["source_ablation"],
        "components": list(entry.get("components", [])),
        "seed": int(args.seed),
        "git_commit": _git_commit(),
        "resolved_config_path": str((run_dir / "resolved_config.yaml").resolve()),
        "resolved_config_sha256": _sha256(run_dir / "resolved_config.yaml"),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": result.get("gpu_name") or (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "dataset_root": config.get("dataloader", {}).get("dataset_root", config.get("processed_dataset", {}).get("root")),
        "dataset_identifier": validation_identity.get("dataset_root") if isinstance(validation_identity, Mapping) else None,
        "dataset_identity": validation_identity,
        "dataset_manifest_hash": validation_identity.get("sample_index_sha256") if isinstance(validation_identity, Mapping) else None,
        "split_identifier": validation_identity.get("split") if isinstance(validation_identity, Mapping) else "val",
        "split_manifest_path": validation_identity.get("sample_index_path") if isinstance(validation_identity, Mapping) else None,
        "split_manifest_hash": validation_identity.get("sample_index_sha256") if isinstance(validation_identity, Mapping) else None,
        "normalization": normalization,
        "normalization_statistics_identifier": str(normalization_path) if normalization_path is not None else None,
        "normalization_hash": _sha256(normalization_path),
        "start_time": started_at.isoformat(),
        "start_timestamp": started_at.isoformat(),
        "end_time": ended_at.isoformat(),
        "end_timestamp": ended_at.isoformat(),
        "run_id": result.get("run_name"),
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_screening_val_loss": result.get("best_val_loss"),
        "stopped_early": result.get("stopped_early"),
        "training_sampling_protocol": result.get("training_sampling_protocol"),
        "efficiency": efficiency,
    }
    _json_write(run_dir / "run_metadata.json", metadata)

    metrics_payload = {
        "schema_version": 1,
        "architecture": args.finalist,
        "source_ablation": entry["source_ablation"],
        "components": list(entry.get("components", [])),
        "seed": int(args.seed),
        "best_epoch": best_epoch,
        "epochs_trained": int(result.get("epochs_completed", len(rows))),
        "stopped_early": bool(result.get("stopped_early", False)),
        "training_sampling_protocol": result.get("training_sampling_protocol"),
        "best_checkpoint": str(best_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "best_screening_validation": best_validation["metrics"],
        "full_validation": full_metrics,
        "full_validation_per_fire": full_result.get("per_fire", {}),
        "full_validation_by_activity_bin": full_result.get("by_activity_bin", {}),
        "efficiency": efficiency,
        **efficiency,
    }
    _json_write(run_dir / "metrics.json", metrics_payload)

    lines = [
        f"Finalist: {args.finalist}",
        f"Source ablation: {entry['source_ablation']}",
        f"Components: {', '.join(str(value) for value in entry.get('components', []))}",
        f"Seed: {args.seed}",
        f"Epochs trained: {metrics_payload['epochs_trained']}",
        f"Best epoch: {best_epoch}",
        f"Best screening validation loss: {_format(result.get('best_val_loss'))}",
        f"Stopped early: {metrics_payload['stopped_early']}",
        "",
        "FULL VALIDATION - BEST CHECKPOINT",
    ]
    for key in (
        "full_val_total_patch_count", "full_val_fire_patch_count", "full_val_no_fire_patch_count",
        "full_val_no_fire_patch_percent", "full_val_dice", "full_val_iou", "full_val_precision", "full_val_recall",
        "full_val_surface_mae", "full_val_surface_rmse", "full_val_canopy_mae", "full_val_canopy_rmse",
        "full_val_energy_log_mae", "full_val_energy_log_rmse", "full_val_energy_mw_mae", "full_val_energy_mw_rmse",
        "full_val_active_surface_mae", "full_val_active_surface_rmse",
        "full_val_active_canopy_mae", "full_val_active_canopy_rmse",
        "full_val_active_energy_log_mae", "full_val_active_energy_log_rmse",
        "full_val_no_fire_mask_prob_mean", "full_val_no_fire_mask_false_positive_rate",
        "full_val_no_fire_patch_false_positive_rate", "full_val_no_fire_surface_pred_mean",
        "full_val_no_fire_canopy_pred_mean", "full_val_no_fire_energy_log_pred_mean",
        "full_val_no_fire_surface_abs_pred_mean", "full_val_no_fire_canopy_abs_pred_mean",
        "full_val_no_fire_energy_log_abs_pred_mean", "full_val_no_fire_surface_mae",
        "full_val_no_fire_canopy_mae", "full_val_no_fire_energy_log_mae",
        "full_val_no_fire_energy_mw_abs_mean",
    ):
        lines.append(f"{key}: {_format(full_metrics.get(key))}")
    lines.extend(["", "EFFICIENCY"])
    lines.extend(f"{key}: {_format(value)}" for key, value in efficiency.items())
    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
