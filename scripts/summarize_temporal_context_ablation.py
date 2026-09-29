#!/usr/bin/env python3
"""Strictly aggregate all nine completed temporal-context ablation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

from scripts.prepare_temporal_context_ablation import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_RESULT_ROOT,
    MODE_ORDER,
    MODE_T,
    SEEDS,
)


CORE_METRICS = {
    "dice": "full_val_dice",
    "iou": "full_val_iou",
    "surface_mae": "full_val_surface_mae",
    "canopy_mae": "full_val_canopy_mae",
    "energy_log_mae": "full_val_energy_log_mae",
    "active_canopy_mae": "full_val_active_canopy_mae",
}
AUXILIARY_METRICS = {
    "precision": "full_val_precision",
    "recall": "full_val_recall",
    "f1": "full_val_f1",
    "surface_rmse": "full_val_surface_rmse",
    "canopy_rmse": "full_val_canopy_rmse",
    "energy_log_rmse": "full_val_energy_log_rmse",
    "active_surface_mae": "full_val_active_surface_mae",
    "active_surface_rmse": "full_val_active_surface_rmse",
    "active_canopy_rmse": "full_val_active_canopy_rmse",
    "active_energy_log_mae": "full_val_active_energy_log_mae",
    "active_energy_log_rmse": "full_val_active_energy_log_rmse",
    "total_patch_count": "full_val_total_patch_count",
    "fire_patch_count": "full_val_fire_patch_count",
    "no_fire_patch_count": "full_val_no_fire_patch_count",
    "no_fire_patch_percent": "full_val_no_fire_patch_percent",
    "no_fire_mask_prob_mean": "full_val_no_fire_mask_prob_mean",
    "no_fire_mask_false_positive_rate": "full_val_no_fire_mask_false_positive_rate",
    "no_fire_patch_false_positive_rate": "full_val_no_fire_patch_false_positive_rate",
    "no_fire_surface_mae": "full_val_no_fire_surface_mae",
    "no_fire_canopy_mae": "full_val_no_fire_canopy_mae",
    "no_fire_energy_log_mae": "full_val_no_fire_energy_log_mae",
}
DISPLAY_NAMES = {
    "dice": "Dice",
    "iou": "IoU",
    "surface_mae": "Surface MAE",
    "canopy_mae": "Canopy MAE",
    "energy_log_mae": "Energy Log MAE",
    "active_canopy_mae": "Active Canopy MAE",
}
MODE_NAMES = {"single": "SINGLE", "sparse5": "SPARSE5", "dense5": "DENSE5"}


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_float(value: Any, *, label: str) -> float:
    if value is None:
        raise ValueError(f"Missing required metric: {label}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite metric {label}: {value!r}")
    return result


def _format_raw(value: float) -> str:
    return f"{value:.8g}"


def _format_summary(values: list[float], scale: float = 1.0) -> str:
    return f"{mean(values) * scale:.6g} ± {stdev(values) * scale:.6g}"


def load_run(result_root: Path, manifest: Mapping[str, Any], mode: str, seed: int) -> dict[str, Any]:
    run_dir = result_root / mode / f"seed{seed}"
    metadata_path = run_dir / "temporal_ablation_metadata.json"
    metrics_path = run_dir / "metrics.json"
    full_path = run_dir / "full_validation_metrics.json"
    history_csv = run_dir / "training_history.csv"
    resolved_config = run_dir / "resolved_config.yaml"
    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    required = [metadata_path, metrics_path, full_path, history_csv, resolved_config]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete temporal ablation run {mode}/seed{seed}; missing: {missing}")
    metadata = _load_json(metadata_path)
    metrics_payload = _load_json(metrics_path)
    full_payload = _load_json(full_path)
    full_metrics = full_payload.get("metrics")
    if not isinstance(full_metrics, Mapping):
        raise ValueError(f"Malformed full-validation payload: {full_path}")
    saved_metrics = metrics_payload.get("full_validation")
    if not isinstance(saved_metrics, Mapping) or dict(saved_metrics) != dict(full_metrics):
        raise ValueError(f"Full-validation metric copies disagree for {mode}/seed{seed}.")
    if metadata.get("input_mode") != mode or int(metadata.get("seed", -1)) != seed:
        raise ValueError(f"Run identity mismatch in {metadata_path}.")
    if int(metadata.get("T", -1)) != MODE_T[mode]:
        raise ValueError(f"Temporal length mismatch in {metadata_path}.")
    if bool(metadata.get("test_split_used", True)):
        raise ValueError(f"Run claims test-split use: {metadata_path}")
    if int(metadata.get("common_validation_sample_count", -1)) != int(manifest["sample_counts"]["val"]):
        raise ValueError(f"Common validation count mismatch in {metadata_path}.")
    if int(full_metrics.get("full_val_total_patch_count", -1)) != int(manifest["sample_counts"]["val"]):
        raise ValueError(f"Full validation is partial for {mode}/seed{seed}.")
    checkpoint_from_metadata = Path(str(metadata.get("best_checkpoint", "")))
    if not checkpoint_from_metadata.is_file() and not best_checkpoint.is_file():
        raise FileNotFoundError(f"Best checkpoint missing for {mode}/seed{seed}.")
    best_epoch = int(metadata.get("best_epoch", -1))
    if not 1 <= best_epoch <= 10:
        raise ValueError(f"Invalid best epoch for {mode}/seed{seed}: {best_epoch}")
    row: dict[str, Any] = {
        "input_mode": mode,
        "seed": seed,
        "T": MODE_T[mode],
        "temporal_offsets": json.dumps(
            metadata.get("fixed_temporal_offsets_minutes")
            if mode != "dense5"
            else {
                "definition": metadata.get("dense_definition"),
                "patterns": metadata.get("dense_offset_pattern_distribution_minutes"),
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        "best_epoch": best_epoch,
    }
    for output_name, source_name in CORE_METRICS.items():
        row[output_name] = _finite_float(full_metrics.get(source_name), label=f"{mode}/seed{seed}/{source_name}")
    for output_name, source_name in AUXILIARY_METRICS.items():
        value = full_metrics.get(source_name)
        row[output_name] = None if value is None else _finite_float(value, label=f"{mode}/seed{seed}/{source_name}")
    row["dice_x100"] = row["dice"] * 100.0
    row["iou_x100"] = row["iou"] * 100.0
    for key in ("surface_mae", "canopy_mae", "energy_log_mae", "active_canopy_mae"):
        row[f"{key}_x1000"] = row[key] * 1000.0
    return row


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fieldnames = list(rows[0])
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def make_text(manifest: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> str:
    grouped = {mode: [row for row in rows if row["input_mode"] == mode] for mode in MODE_ORDER}
    for mode, values in grouped.items():
        if [int(row["seed"]) for row in values] != list(SEEDS):
            raise RuntimeError(f"Unexpected seed inventory for {mode}: {[row['seed'] for row in values]}")
    lines = [
        "=" * 60,
        "TEMPORAL CONTEXT ABLATION",
        "=" * 60,
        "",
        "PROTOCOL",
        "- architecture: original FLARE / CAWFE-Latte baseline (baseline_cnn, baseline temporal pooling, shared decoder)",
        f"- train fires ({len(manifest['train_fires'])}): {', '.join(manifest['train_fires'])}",
        f"- validation fires ({len(manifest['validation_fires'])}): {', '.join(manifest['validation_fires'])}",
        f"- prediction horizon: +{manifest['target_horizon_minutes']} minutes from the common latest input",
        f"- training budget: exactly {manifest['epoch_cap']} epochs; 7500 train batches per computational epoch; batch size 8",
        "- selection/evaluation: canonical fixed stratified validation screening; deterministic full validation of best checkpoint",
        "- normalization: 129-channel train-only statistics fitted separately per mode; validation/test excluded",
        f"- common train samples: {manifest['sample_counts']['train']}",
        f"- common validation samples: {manifest['sample_counts']['val']}",
        "- held-out test use: none",
        "",
        "TEMPORAL DEFINITIONS",
        "Single: [t] (T=1).",
        "Sparse5: [t-40, t-30, t-20, t-10, t] minutes (T=5; exact tensor-minute matching).",
        "Dense5: four immediately preceding available native states plus t (T=5).",
        f"Dense5 train five-state span distribution: {manifest['timing_audit']['dense_span_minutes']['train']}",
        f"Dense5 validation five-state span distribution: {manifest['timing_audit']['dense_span_minutes']['val']}",
        f"Dense5 train offset-pattern distribution: {manifest['timing_audit']['dense_offset_patterns_minutes']['train']}",
        f"Dense5 validation offset-pattern distribution: {manifest['timing_audit']['dense_offset_patterns_minutes']['val']}",
        "",
        "PER-SEED RESULTS (raw, unscaled)",
        "Configuration | Seed | Dice | IoU | Surface MAE | Canopy MAE | Energy Log MAE | Active Canopy MAE",
    ]
    for row in rows:
        values = " | ".join(_format_raw(float(row[key])) for key in CORE_METRICS)
        lines.append(f"{MODE_NAMES[str(row['input_mode'])]} | {row['seed']} | {values}")
    lines.extend(
        [
            "",
            "MEAN ± STD (raw, sample standard deviation across 3 seeds)",
            "Configuration | Dice | IoU | Surface MAE | Canopy MAE | Energy Log MAE | Active Canopy MAE",
        ]
    )
    for mode in MODE_ORDER:
        summaries = [_format_summary([float(row[key]) for row in grouped[mode]]) for key in CORE_METRICS]
        lines.append(f"{MODE_NAMES[mode]} | " + " | ".join(summaries))
    lines.extend(
        [
            "",
            "MEAN ± STD (paper scale: Dice/IoU ×10^2; MAEs ×10^3)",
            "Configuration | Dice ×10^2 | IoU ×10^2 | Surface MAE ×10^3 | Canopy MAE ×10^3 | Energy Log MAE ×10^3 | Active Canopy MAE ×10^3",
        ]
    )
    for mode in MODE_ORDER:
        summaries = []
        for key in CORE_METRICS:
            scale = 100.0 if key in {"dice", "iou"} else 1000.0
            summaries.append(_format_summary([float(row[key]) for row in grouped[mode]], scale))
        lines.append(f"{MODE_NAMES[mode]} | " + " | ".join(summaries))
    dense_means = {key: mean(float(row[key]) for row in grouped["dense5"]) for key in CORE_METRICS}
    lines.extend(
        [
            "",
            "RELATIVE COMPARISON (aggregate means; reference = DENSE5)",
            "Positive signed Dice/IoU delta means higher than DENSE5; positive MAE percentage means worse than DENSE5.",
        ]
    )
    for mode in ("single", "sparse5"):
        mode_means = {key: mean(float(row[key]) for row in grouped[mode]) for key in CORE_METRICS}
        lines.append(f"{MODE_NAMES[mode]} relative to DENSE5:")
        for key in ("dice", "iou"):
            delta = mode_means[key] - dense_means[key]
            lines.append(f"- {DISPLAY_NAMES[key]} difference: {delta:+.8g} (absolute magnitude {abs(delta):.8g})")
        for key in ("surface_mae", "canopy_mae", "energy_log_mae", "active_canopy_mae"):
            reference = dense_means[key]
            if reference == 0.0:
                lines.append(f"- {DISPLAY_NAMES[key]} percentage difference: undefined (DENSE5 mean is zero)")
            else:
                percent = 100.0 * (mode_means[key] - reference) / reference
                lines.append(f"- {DISPLAY_NAMES[key]} percentage difference: {percent:+.6g}%")
    lines.extend(
        [
            "",
            "Interpretation: higher Dice/IoU is better; lower MAE is better.",
            "Raw values are retained in this report and temporal_context_summary.csv.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize(artifact_root: Path, result_root: Path) -> tuple[Path, Path]:
    manifest_path = artifact_root / "preparation_manifest.json"
    manifest = _load_json(manifest_path)
    if not bool(manifest.get("ready_for_training", False)):
        raise RuntimeError(f"Preparation is not training-ready: {manifest_path}")
    if int(manifest.get("epoch_cap", -1)) != 10:
        raise RuntimeError("Preparation manifest does not specify the required 10-epoch cap.")
    rows = [load_run(result_root, manifest, mode, seed) for mode in MODE_ORDER for seed in SEEDS]
    if len(rows) != 9:
        raise RuntimeError(f"Expected exactly 9 runs, got {len(rows)}.")
    csv_path = result_root / "temporal_context_summary.csv"
    text_path = result_root / "temporal_context_summary.txt"
    write_csv(csv_path, rows)
    _atomic_write(text_path, make_text(manifest, rows))
    print(text_path)
    print(csv_path)
    return text_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args()
    summarize(args.artifact_root.expanduser().resolve(), args.result_root.expanduser().resolve())


if __name__ == "__main__":
    main()
