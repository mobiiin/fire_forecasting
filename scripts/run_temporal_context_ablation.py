#!/usr/bin/env python3
"""Train one prepared FLARE temporal-context ablation run from scratch."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from typing import Any, Mapping

import torch

from scripts.prepare_temporal_context_ablation import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_RESULT_ROOT,
    FIXED_OFFSETS_MINUTES,
    MODE_ORDER,
    MODE_T,
    SEEDS,
    TARGET_HORIZON_MINUTES,
    _assert_original_baseline,
    sha256_file,
)
from src.config import load_config
from src.models.model_factory import build_model_from_config
from src.training.train import train_model_from_config


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
        )
    except Exception:
        return None
    return result.stdout.strip() or None


REQUIRED_FULL_METRICS = (
    "full_val_dice",
    "full_val_iou",
    "full_val_surface_mae",
    "full_val_canopy_mae",
    "full_val_energy_log_mae",
    "full_val_active_canopy_mae",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_history(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate_prepared_config(config: Mapping[str, Any], manifest: Mapping[str, Any], mode: str) -> None:
    _assert_original_baseline(config)
    training = config.get("training", {})
    validation = training.get("validation", {})
    screening = validation.get("screening", {})
    full = validation.get("full", {})
    problems: list[str] = []
    if not bool(manifest.get("ready_for_training", False)):
        problems.append("preparation manifest is not marked ready_for_training")
    if int(config.get("input_sequence_length", -1)) != MODE_T[mode]:
        problems.append("input_sequence_length does not match mode")
    if int(training.get("max_epochs", -1)) != 10 or int(training.get("epochs", -1)) != 10:
        problems.append("max_epochs and epochs must both be exactly 10")
    if int(training.get("max_train_batches", -1)) != 7500:
        problems.append("max_train_batches must be 7500")
    if training.get("max_train_batches_per_epoch") not in (None, 0, "null"):
        problems.append("max_train_batches_per_epoch must be disabled")
    if int(training.get("batch_size", -1)) != 8:
        problems.append("batch_size must be 8")
    if int(training.get("gradient_accumulation_steps", -1)) != 1:
        problems.append("gradient_accumulation_steps must be 1")
    if bool(training.get("run_test_after_training", False)) or bool(training.get("run_external_test_after_training", False)):
        problems.append("test evaluation must be disabled")
    if bool(config.get("dataloader", {}).get("include_test_split", True)):
        problems.append("test split construction must be disabled")
    if config.get("evaluation", {}).get("split") != "val":
        problems.append("evaluation split must be val")
    if not bool(screening.get("enabled", False)) or screening.get("sampling") != "stratified_fixed":
        problems.append("canonical stratified screening validation is not enabled")
    if not bool(full.get("enabled", False)) or full.get("checkpoint") != "best":
        problems.append("best-checkpoint full validation is not enabled")
    sample_index = Path(str(config.get("dataloader", {}).get("sample_index_path", "")))
    expected_index = manifest.get("indices", {}).get("modes", {}).get(mode, {})
    if not sample_index.is_file() or sha256_file(sample_index) != expected_index.get("sha256"):
        problems.append("prepared temporal sample index is missing or has changed")
    normalization_path = Path(str(config.get("normalization", {}).get("stats_path", "")))
    expected_normalization = manifest.get("normalization", {}).get(mode, {})
    if not normalization_path.is_file() or sha256_file(normalization_path) != expected_normalization.get("json_sha256"):
        problems.append("prepared normalization metadata is missing or has changed")
    else:
        normalization_payload = json.loads(normalization_path.read_text(encoding="utf-8"))
        if normalization_payload.get("fit_split") != "train":
            problems.append("prepared normalization metadata is not marked train-only")
        if int(normalization_payload.get("input_channels", -1)) != 129:
            problems.append("prepared normalization metadata does not describe 129 channels")
    normalization_npz = Path(str(expected_normalization.get("npz_path", "")))
    if not normalization_npz.is_file() or sha256_file(normalization_npz) != expected_normalization.get("npz_sha256"):
        problems.append("prepared normalization numeric archive is missing or has changed")
    if config.get("normalization", {}).get("fit_split") != "train":
        problems.append("normalization fit_split must be train")
    if problems:
        raise ValueError("Unsafe temporal-context config:\n- " + "\n- ".join(problems))


def _parameter_count(config: Mapping[str, Any]) -> dict[str, int]:
    model = build_model_from_config(config, input_channels=129)
    before = int(sum(parameter.numel() for parameter in model.parameters()))
    model.alignment._spatial_position(64, 64, device=torch.device("cpu"), dtype=torch.float32)
    after = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    return {"before_lazy_spatial_position": before, "with_64x64_spatial_position": after, "trainable": trainable}


def run(mode: str, seed: int, *, artifact_root: Path, result_root: Path) -> Path:
    if mode not in MODE_ORDER:
        raise ValueError(f"Unknown input mode {mode!r}; expected one of {MODE_ORDER}.")
    if int(seed) not in SEEDS:
        raise ValueError(f"Unknown seed {seed}; expected one of {SEEDS}.")
    manifest_path = artifact_root / "preparation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Preparation manifest is missing: {manifest_path}. Run 00_prepare_temporal_ablation.slurm first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = artifact_root / mode / "config_resolved.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Prepared config is missing: {config_path}")
    expected_config_hash = manifest.get("configs", {}).get(mode, {}).get("sha256")
    if sha256_file(config_path) != expected_config_hash:
        raise RuntimeError(f"Prepared config hash changed: {config_path}")
    config = load_config(config_path)
    _validate_prepared_config(config, manifest, mode)

    expected_run_dir = result_root / mode / f"seed{int(seed)}"
    if expected_run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists; refusing an accidental collision/resume: {expected_run_dir}"
        )
    training = dict(config["training"])
    training["seed"] = int(seed)
    training["max_epochs"] = 10
    training["epochs"] = 10
    training["run_name"] = f"seed{int(seed)}"
    training["overwrite_run"] = False
    output = dict(training["output"])
    output["root_dir"] = str((result_root / mode).resolve())
    output["flat_run_layout"] = True
    training["output"] = output
    config["training"] = training
    config["seed"] = int(seed)
    checkpoint = dict(config.get("checkpoint", {}))
    checkpoint["resume"] = False
    config["checkpoint"] = checkpoint
    _validate_prepared_config(config, manifest, mode)

    counts = _parameter_count(config)
    prepared_counts = manifest["sanity_checks"]["parameter_counts"][mode]
    if counts["with_64x64_spatial_position"] != int(prepared_counts["with_64x64_spatial_position"]):
        raise RuntimeError("Model parameter count no longer matches preparation.")
    print(f"Mode: {mode}")
    print(f"Seed: {seed}")
    print(f"Config: {config_path}")
    print(f"Output: {expected_run_dir}")
    print(f"T: {MODE_T[mode]}")
    print(f"Target horizon: +{TARGET_HORIZON_MINUTES} minutes")
    print(f"Parameters (64x64): {counts['with_64x64_spatial_position']}")
    print("Training from scratch for exactly 10 epochs; test split is disabled.")

    started_at = datetime.now(timezone.utc)
    result = train_model_from_config(config)
    ended_at = datetime.now(timezone.utc)
    run_dir = Path(str(result["run_dir"])).resolve()
    if run_dir != expected_run_dir.resolve():
        raise RuntimeError(f"Unexpected run directory: {run_dir}; expected {expected_run_dir.resolve()}")
    history = [dict(row) for row in result.get("history_rows", [])]
    if len(history) != 10 or int(result.get("epochs_completed", len(history))) != 10:
        raise RuntimeError(f"Run did not complete exactly 10 epochs: rows={len(history)} result={result.get('epochs_completed')}")
    best_epoch = int(result["best_epoch"])
    if not 1 <= best_epoch <= 10:
        raise RuntimeError(f"Invalid best epoch: {best_epoch}")
    best_checkpoint = Path(str(result["best_checkpoint_path"]))
    if not best_checkpoint.is_file():
        raise RuntimeError(f"Best checkpoint is missing: {best_checkpoint}")
    full_result = result.get("full_validation", {})
    metrics = full_result.get("metrics", {}) if isinstance(full_result, Mapping) else {}
    missing = [name for name in REQUIRED_FULL_METRICS if metrics.get(name) is None]
    if missing:
        raise RuntimeError(f"Full validation is missing paper metrics: {missing}")
    if int(metrics.get("full_val_total_patch_count", -1)) != int(manifest["sample_counts"]["val"]):
        raise RuntimeError("Full validation did not evaluate every shared validation sample exactly once.")

    resolved_source = run_dir / "configs/resolved_config.yaml"
    if not resolved_source.is_file():
        raise RuntimeError(f"Resolved run config is missing: {resolved_source}")
    shutil.copyfile(resolved_source, run_dir / "resolved_config.yaml")
    _write_history(run_dir / "training_history.csv", history)
    _write_json(run_dir / "training_history.json", {"rows": history})
    metadata = {
        "schema_version": 1,
        "input_mode": mode,
        "seed": int(seed),
        "T": MODE_T[mode],
        "fixed_temporal_offsets_minutes": FIXED_OFFSETS_MINUTES.get(mode),
        "dense_definition": "four immediately preceding available native CAWFE states plus latest" if mode == "dense5" else None,
        "dense_span_distribution_minutes": manifest["timing_audit"]["dense_span_minutes"] if mode == "dense5" else None,
        "dense_offset_pattern_distribution_minutes": manifest["timing_audit"]["dense_offset_patterns_minutes"] if mode == "dense5" else None,
        "target_horizon_minutes": TARGET_HORIZON_MINUTES,
        "time_coordinate_source": manifest["time_coordinate_source"],
        "architecture": "original_flare_cawfe_latte_baseline",
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "epochs_trained": len(history),
        "best_screening_val_loss": result.get("best_val_loss"),
        "parameter_counts": counts,
        "train_fires": manifest["train_fires"],
        "validation_fires": manifest["validation_fires"],
        "train_fire_split_identifier": manifest["train_fires"],
        "validation_fire_split_identifier": manifest["validation_fires"],
        "test_split_used": False,
        "common_train_sample_count": int(manifest["sample_counts"]["train"]),
        "common_validation_sample_count": int(manifest["sample_counts"]["val"]),
        "sample_index": manifest["indices"]["modes"][mode],
        "normalization": manifest["normalization"][mode],
        "prepared_config_path": str(config_path.resolve()),
        "prepared_config_sha256": sha256_file(config_path),
        "resolved_config_path": str((run_dir / "resolved_config.yaml").resolve()),
        "resolved_config_sha256": sha256_file(run_dir / "resolved_config.yaml"),
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_checkpoint.resolve()),
        "full_validation_metric_path": str((run_dir / "full_validation_metrics.json").resolve()),
    }
    _write_json(run_dir / "temporal_ablation_metadata.json", metadata)
    _write_json(
        run_dir / "metrics.json",
        {
            "schema_version": 1,
            "input_mode": mode,
            "seed": int(seed),
            "T": MODE_T[mode],
            "best_epoch": best_epoch,
            "best_checkpoint": str(best_checkpoint.resolve()),
            "full_validation": dict(metrics),
        },
    )
    print(f"Completed: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODE_ORDER)
    parser.add_argument("seed", type=int, choices=SEEDS)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--print-output", action="store_true")
    args = parser.parse_args()
    if args.print_config:
        print((args.artifact_root / args.mode / "config_resolved.yaml").resolve())
        return
    if args.print_output:
        print((args.result_root / args.mode / f"seed{args.seed}").resolve())
        return
    run(args.mode, args.seed, artifact_root=args.artifact_root.resolve(), result_root=args.result_root.resolve())


if __name__ == "__main__":
    main()
