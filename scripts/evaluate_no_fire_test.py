#!/usr/bin/env python3
"""Evaluate frozen CAWFE-Latte finalists on the canonical no-fire test subset.

This script never trains or selects a checkpoint. It reuses complete locked-test
artifacts when they contain the exact paper quantities; otherwise it invokes the
existing held-out evaluator on the already validation-selected best checkpoint.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from scripts.run_table2_baseline import (
    _complete_test_artifact,
    evaluate_existing_cawfe_run,
    evaluate_learned_run,
)
from scripts.summarize_table2_results import _discover_cawfe_training_run
from src.config import load_config
from src.evaluation.fire_activity import (
    ACTIVE_FRACTION_THRESHOLD,
    FIRE_MASK_CHANNEL,
    FIRE_MASK_THRESHOLD,
    PREDICTED_FIRE_THRESHOLD,
)
from src.evaluation.table2_protocol import table2_protocol_identity


MODELS = ("baseline", "GA_Q2")
SEEDS = (42, 123, 2026)
MODEL_TO_TABLE2 = {
    "baseline": "cawfe_latte_baseline",
    "GA_Q2": "cawfe_latte_final",
}
DISPLAY_NAMES = {
    "baseline": "Baseline",
    "GA_Q2": "CAWFE-Latte",
}
CAWFE_ROOT = Path("artifacts/final_training/cawfe_latte")
OUTPUT_ROOT = Path("artifacts/no_fire_test_analysis")

EXACT_METRIC_KEYS = (
    "test_total_patch_count",
    "test_fire_patch_count",
    "test_no_fire_patch_count",
    "test_no_fire_pixel_count",
    "test_no_fire_mask_prob_mean",
    "test_no_fire_mask_false_positive_rate",
    "test_no_fire_patch_false_positive_rate",
    "test_no_fire_surface_pred_mean",
    "test_no_fire_surface_abs_pred_mean",
    "test_no_fire_surface_mae",
    "test_no_fire_canopy_pred_mean",
    "test_no_fire_canopy_abs_pred_mean",
    "test_no_fire_canopy_mae",
    "test_no_fire_energy_log_pred_mean",
    "test_no_fire_energy_log_abs_pred_mean",
    "test_no_fire_energy_log_mae",
)
SAMPLE_REQUIRED_COLUMNS = (
    "sample_id",
    "fire_name",
    "is_no_fire",
    "target_active_fraction",
    "predicted_fire_fraction",
    "mean_mask_probability",
    "mean_surface_prediction",
    "mean_canopy_prediction",
    "mean_energy_log_prediction",
    "no_fire_mask_false_positive_rate",
    "no_fire_surface_abs_pred_mean",
    "no_fire_canopy_abs_pred_mean",
    "no_fire_energy_log_abs_pred_mean",
)
MAIN_METRICS = {
    "mask_fp": "mask_fp",
    "surface_abs_mean": "surface_abs_mean",
    "canopy_abs_mean": "canopy_abs_mean",
    "energy_abs_mean": "energy_log_abs_mean",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    columns = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    if not columns:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _resolved_config_path(run_dir: Path) -> Path:
    for candidate in (run_dir / "resolved_config.yaml", run_dir / "configs" / "resolved_config.yaml"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Resolved config is missing for frozen run: {run_dir}")


def _checkpoint_path(run_dir: Path) -> Path:
    checkpoint = run_dir / "checkpoints" / "best_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Validation-selected best checkpoint is missing: {checkpoint}")
    return checkpoint.resolve()


def discover_frozen_run(model: str, seed: int, cawfe_root: Path = CAWFE_ROOT) -> Path:
    run_dir = _discover_cawfe_training_run(model, int(seed), cawfe_root)
    if run_dir is None:
        raise FileNotFoundError(
            f"No completed full-training run with a best validation checkpoint was found for "
            f"{model} seed={seed} beneath {cawfe_root}."
        )
    checkpoint = _checkpoint_path(run_dir)
    config = load_config(_resolved_config_path(run_dir))
    if str(config.get("final_training", {}).get("finalist")) != model:
        raise RuntimeError(f"Finalist identity mismatch in {run_dir}")
    if int(config.get("training", {}).get("seed", -1)) != int(seed):
        raise RuntimeError(f"Seed mismatch in {run_dir}")
    if checkpoint.parent.parent != run_dir.resolve():
        raise RuntimeError(f"Checkpoint escaped its frozen run directory: {checkpoint}")
    return run_dir.resolve()


def _sample_metrics_path(run_dir: Path, artifact: Mapping[str, Any]) -> Path:
    evaluation = run_dir / "evaluation"
    csv_path = evaluation / "test_sample_metrics.csv"
    if csv_path.is_file():
        return csv_path
    configured = artifact.get("sample_metrics_path")
    if configured:
        candidate = Path(str(configured)).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    parquet_path = evaluation / "test_sample_metrics.parquet"
    if parquet_path.is_file():
        return parquet_path
    raise FileNotFoundError(f"Complete per-sample test metrics are missing under {evaluation}")


def _read_sample_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                f"Reading {path} requires pandas plus a Parquet engine; the locked evaluator normally also writes CSV."
            ) from exc
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported sample-metric format: {path}")


def _load_artifact(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "evaluation" / "test_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite: {value!r}")
    return result


def _int(value: Any, name: str) -> int:
    result = int(_float(value, name))
    if float(result) != float(value):
        raise ValueError(f"{name} is not an integer: {value!r}")
    return result


def _optional_float(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None or value == "":
        return None
    return _float(value, key)


def _ids_hash(sample_ids: Iterable[str]) -> str:
    ordered = sorted(str(sample_id) for sample_id in sample_ids)
    encoded = ("\n".join(ordered) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_has_exact_metrics(run_dir: Path) -> tuple[bool, str]:
    if not _complete_test_artifact(run_dir):
        return False, "locked test artifact is absent or incomplete"
    try:
        artifact = _load_artifact(run_dir)
        metrics = artifact["metrics"]
        missing = [key for key in EXACT_METRIC_KEYS if metrics.get(key) is None]
        if missing:
            return False, "exact metrics missing: " + ", ".join(missing)
        checkpoint = _checkpoint_path(run_dir)
        artifact_checkpoint = artifact.get("checkpoint")
        if not artifact_checkpoint or Path(str(artifact_checkpoint)).expanduser().resolve() != checkpoint:
            return False, "test artifact is not tied to the validation-selected best_model.pt"
        sample_path = _sample_metrics_path(run_dir, artifact)
        rows = _read_sample_rows(sample_path)
        if not rows:
            return False, "per-sample test artifact is empty"
        missing_columns = [key for key in SAMPLE_REQUIRED_COLUMNS if key not in rows[0]]
        if missing_columns:
            return False, "per-sample exact fields missing: " + ", ".join(missing_columns)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return False, str(exc)
    return True, "complete exact no-fire metrics already exist"


def ensure_exact_test_artifact(model: str, seed: int, run_dir: Path) -> dict[str, Any]:
    exact, reason = _artifact_has_exact_metrics(run_dir)
    if exact:
        print(f"REUSING COMPLETE EXACT TEST ARTIFACT: {model} seed={seed} | {run_dir}")
        return _load_artifact(run_dir)

    print(f"EXACT NO-FIRE TEST EVALUATION REQUIRED: {model} seed={seed} | {reason}")
    table2_name = MODEL_TO_TABLE2[model]
    if _complete_test_artifact(run_dir):
        # The generic complete artifact predates the direct-absolute schema.
        # Re-run only locked inference; checkpoint selection remains untouched.
        config = load_config(_resolved_config_path(run_dir))
        identity = table2_protocol_identity(config, table2_name, int(seed))
        evaluate_learned_run(
            run_dir=run_dir,
            config=config,
            baseline=table2_name,
            seed=int(seed),
            identity=identity,
            require_full_validation=False,
        )
    else:
        evaluate_existing_cawfe_run(run_dir, table2_name, int(seed))

    exact, reason = _artifact_has_exact_metrics(run_dir)
    if not exact:
        raise RuntimeError(f"Locked evaluation did not produce an exact reusable artifact: {reason}")
    return _load_artifact(run_dir)


def _weighted_mean(rows: Sequence[Mapping[str, Any]], key: str, weights: Sequence[int]) -> float:
    values = [_float(row[key], key) for row in rows]
    denominator = sum(weights)
    if denominator <= 0:
        raise RuntimeError(f"No pixels available for weighted metric {key}")
    return math.fsum(value * weight for value, weight in zip(values, weights)) / denominator


def _validate_close(actual: float, expected: float, name: str, *, tolerance: float = 2e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(f"Per-sample and streaming {name} disagree: {actual} vs {expected}")


def extract_record(
    model: str,
    seed: int,
    run_dir: Path,
    artifact: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[str]]:
    if artifact is None:
        exact, reason = _artifact_has_exact_metrics(run_dir)
        if not exact:
            raise RuntimeError(f"Cannot aggregate {model} seed={seed}: {reason}")
        artifact = _load_artifact(run_dir)
    metrics = artifact["metrics"]
    sample_path = _sample_metrics_path(run_dir, artifact)
    sample_rows = _read_sample_rows(sample_path)

    total = _int(artifact["dataset_sample_count"], "dataset_sample_count")
    evaluated = _int(artifact["evaluated_test_samples"], "evaluated_test_samples")
    unique_count = _int(artifact["unique_sample_id_count"], "unique_sample_id_count")
    fire_count = _int(artifact["fire_count"], "fire_count")
    no_fire_count = _int(artifact["no_fire_count"], "no_fire_count")
    if not (total == evaluated == unique_count == len(sample_rows)):
        raise RuntimeError(
            f"Incomplete test rows for {model} seed={seed}: total={total}, evaluated={evaluated}, "
            f"unique={unique_count}, rows={len(sample_rows)}"
        )
    if fire_count + no_fire_count != total:
        raise RuntimeError("fire_count + no_fire_count does not equal total test patches")

    all_ids = [str(row["sample_id"]) for row in sample_rows]
    if len(set(all_ids)) != total:
        raise RuntimeError(f"Duplicate held-out sample IDs for {model} seed={seed}")
    no_fire_rows: list[dict[str, Any]] = []
    for row in sample_rows:
        is_no_fire = _int(row["is_no_fire"], "is_no_fire")
        fraction = _float(row["target_active_fraction"], "target_active_fraction")
        if is_no_fire not in (0, 1):
            raise RuntimeError(f"Invalid is_no_fire value: {is_no_fire}")
        expected_no_fire = (
            fraction <= 0.0
            if ACTIVE_FRACTION_THRESHOLD <= 0.0
            else fraction < ACTIVE_FRACTION_THRESHOLD
        )
        if expected_no_fire != bool(is_no_fire):
            raise RuntimeError(
                f"Saved membership violates canonical target-only classification for sample {row['sample_id']}"
            )
        if is_no_fire:
            no_fire_rows.append(row)
    if len(no_fire_rows) != no_fire_count or no_fire_count <= 0:
        raise RuntimeError(
            f"No-fire row mismatch for {model} seed={seed}: artifact={no_fire_count}, rows={len(no_fire_rows)}"
        )

    no_fire_pixels = _int(metrics["test_no_fire_pixel_count"], "test_no_fire_pixel_count")
    if no_fire_pixels <= 0 or no_fire_pixels % no_fire_count:
        raise RuntimeError("No-fire pixel count is not a positive fixed-patch multiple")
    default_pixels = no_fire_pixels // no_fire_count
    weights = [_int(row.get("pixel_count", default_pixels), "pixel_count") for row in no_fire_rows]
    if sum(weights) != no_fire_pixels:
        raise RuntimeError("Per-sample no-fire pixel counts do not equal the streaming pixel count")

    sample_checks = {
        "pixel Mask FP": (
            _weighted_mean(no_fire_rows, "no_fire_mask_false_positive_rate", weights),
            _float(metrics["test_no_fire_mask_false_positive_rate"], "test_no_fire_mask_false_positive_rate"),
        ),
        "mask probability mean": (
            _weighted_mean(no_fire_rows, "mean_mask_probability", weights),
            _float(metrics["test_no_fire_mask_prob_mean"], "test_no_fire_mask_prob_mean"),
        ),
        "surface absolute mean": (
            _weighted_mean(no_fire_rows, "no_fire_surface_abs_pred_mean", weights),
            _float(metrics["test_no_fire_surface_abs_pred_mean"], "test_no_fire_surface_abs_pred_mean"),
        ),
        "canopy absolute mean": (
            _weighted_mean(no_fire_rows, "no_fire_canopy_abs_pred_mean", weights),
            _float(metrics["test_no_fire_canopy_abs_pred_mean"], "test_no_fire_canopy_abs_pred_mean"),
        ),
        "energy-log absolute mean": (
            _weighted_mean(no_fire_rows, "no_fire_energy_log_abs_pred_mean", weights),
            _float(metrics["test_no_fire_energy_log_abs_pred_mean"], "test_no_fire_energy_log_abs_pred_mean"),
        ),
    }
    for name, (actual, expected) in sample_checks.items():
        _validate_close(actual, expected, name)

    per_fire: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in no_fire_rows:
        grouped[str(row["fire_name"])].append(row)
    for fire_name, rows in sorted(grouped.items()):
        fire_weights = [_int(row.get("pixel_count", default_pixels), "pixel_count") for row in rows]
        patch_fp_values = [
            _int(row["no_fire_patch_false_positive"], "no_fire_patch_false_positive")
            if row.get("no_fire_patch_false_positive") not in (None, "")
            else int(_float(row["predicted_fire_fraction"], "predicted_fire_fraction") > 0.0)
            for row in rows
        ]
        per_fire.append(
            {
                "model": model,
                "display_name": DISPLAY_NAMES[model],
                "seed": int(seed),
                "fire_name": fire_name,
                "no_fire_patch_count": len(rows),
                "no_fire_pixel_count": sum(fire_weights),
                "mask_fp": _weighted_mean(rows, "no_fire_mask_false_positive_rate", fire_weights),
                "surface_abs_mean": _weighted_mean(rows, "no_fire_surface_abs_pred_mean", fire_weights),
                "canopy_abs_mean": _weighted_mean(rows, "no_fire_canopy_abs_pred_mean", fire_weights),
                "energy_log_abs_mean": _weighted_mean(rows, "no_fire_energy_log_abs_pred_mean", fire_weights),
                "mask_probability_mean": _weighted_mean(rows, "mean_mask_probability", fire_weights),
                "patch_fp_rate": statistics.fmean(patch_fp_values),
            }
        )
    if sum(int(row["no_fire_patch_count"]) for row in per_fire) != no_fire_count:
        raise RuntimeError("Per-fire no-fire patch counts do not sum to the overall no-fire count")

    checkpoint = _checkpoint_path(run_dir)
    artifact_checkpoint = Path(str(artifact["checkpoint"])).expanduser().resolve()
    if checkpoint != artifact_checkpoint:
        raise RuntimeError(f"Artifact checkpoint mismatch: {artifact_checkpoint} != {checkpoint}")
    no_fire_ids = [str(row["sample_id"]) for row in no_fire_rows]
    record = {
        "model": model,
        "display_name": DISPLAY_NAMES[model],
        "seed": int(seed),
        "checkpoint_path": str(checkpoint),
        "checkpoint_epoch": artifact.get("checkpoint_epoch"),
        "run_dir": str(run_dir),
        "test_metrics_path": str((run_dir / "evaluation" / "test_metrics.json").resolve()),
        "test_sample_metrics_path": str(sample_path.resolve()),
        "metric_scope": artifact.get("metric_scope"),
        "test_used_for_model_selection": artifact.get("test_used_for_model_selection"),
        "total_test_patches": total,
        "fire_test_patches": fire_count,
        "no_fire_test_patches": no_fire_count,
        "no_fire_percentage": 100.0 * no_fire_count / total,
        "no_fire_pixel_count": no_fire_pixels,
        "mask_fp": _float(metrics["test_no_fire_mask_false_positive_rate"], "mask_fp"),
        "surface_abs_mean": _float(metrics["test_no_fire_surface_abs_pred_mean"], "surface_abs_mean"),
        "canopy_abs_mean": _float(metrics["test_no_fire_canopy_abs_pred_mean"], "canopy_abs_mean"),
        "energy_log_abs_mean": _float(metrics["test_no_fire_energy_log_abs_pred_mean"], "energy_log_abs_mean"),
        "mask_probability_mean": _float(metrics["test_no_fire_mask_prob_mean"], "mask_probability_mean"),
        "patch_fp_rate": _float(metrics["test_no_fire_patch_false_positive_rate"], "patch_fp_rate"),
        "surface_signed_mean": _float(metrics["test_no_fire_surface_pred_mean"], "surface_signed_mean"),
        "canopy_signed_mean": _float(metrics["test_no_fire_canopy_pred_mean"], "canopy_signed_mean"),
        "energy_log_signed_mean": _float(metrics["test_no_fire_energy_log_pred_mean"], "energy_log_signed_mean"),
        "surface_mae": _float(metrics["test_no_fire_surface_mae"], "surface_mae"),
        "canopy_mae": _float(metrics["test_no_fire_canopy_mae"], "canopy_mae"),
        "energy_log_mae": _float(metrics["test_no_fire_energy_log_mae"], "energy_log_mae"),
        "surface_abs_mean_minus_mae": _float(metrics["test_no_fire_surface_abs_pred_mean"], "surface_abs_mean")
        - _float(metrics["test_no_fire_surface_mae"], "surface_mae"),
        "canopy_abs_mean_minus_mae": _float(metrics["test_no_fire_canopy_abs_pred_mean"], "canopy_abs_mean")
        - _float(metrics["test_no_fire_canopy_mae"], "canopy_mae"),
        "energy_log_abs_mean_minus_mae": _float(metrics["test_no_fire_energy_log_abs_pred_mean"], "energy_log_abs_mean")
        - _float(metrics["test_no_fire_energy_log_mae"], "energy_log_mae"),
        "surface_rmse": _optional_float(metrics, "test_no_fire_surface_pred_rmse"),
        "canopy_rmse": _optional_float(metrics, "test_no_fire_canopy_pred_rmse"),
        "energy_log_rmse": _optional_float(metrics, "test_no_fire_energy_log_pred_rmse"),
        "surface_target_abs_mean": _optional_float(metrics, "test_no_fire_surface_target_abs_mean"),
        "canopy_target_abs_mean": _optional_float(metrics, "test_no_fire_canopy_target_abs_mean"),
        "energy_log_target_abs_mean": _optional_float(metrics, "test_no_fire_energy_log_target_abs_mean"),
        "full_test_sample_ids_hash": _ids_hash(all_ids),
        "no_fire_subset_hash": _ids_hash(no_fire_ids),
    }
    return record, per_fire, all_ids, no_fire_ids


def _write_worker_record(output_root: Path, record: Mapping[str, Any], per_fire: Sequence[Mapping[str, Any]]) -> None:
    model = str(record["model"])
    seed = int(record["seed"])
    path = output_root / "runs" / model / f"seed_{seed}.json"
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "created_at": _utc_now(),
            "record": dict(record),
            "per_fire": [dict(row) for row in per_fire],
        },
    )


def evaluate_selection(
    models: Sequence[str],
    seeds: Sequence[int],
    *,
    cawfe_root: Path = CAWFE_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> None:
    for model in models:
        for seed in seeds:
            run_dir = discover_frozen_run(model, seed, cawfe_root)
            artifact = ensure_exact_test_artifact(model, seed, run_dir)
            record, per_fire, _, _ = extract_record(model, seed, run_dir, artifact)
            _write_worker_record(output_root, record, per_fire)
            print(
                f"READY: {model} seed={seed} | total={record['total_test_patches']} "
                f"no_fire={record['no_fire_test_patches']} | hash={record['no_fire_subset_hash']}"
            )


def _aggregate_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_model = {model: [record for record in records if record["model"] == model] for model in MODELS}
    for model in MODELS:
        model_records = sorted(by_model[model], key=lambda row: int(row["seed"]))
        actual_seeds = [int(row["seed"]) for row in model_records]
        if actual_seeds != list(SEEDS):
            raise RuntimeError(f"{model} requires seeds {list(SEEDS)}, found {actual_seeds}")
        row: dict[str, Any] = {
            "model": model,
            "display_name": DISPLAY_NAMES[model],
            "seeds": ",".join(str(seed) for seed in actual_seeds),
        }
        for output_prefix, record_key in MAIN_METRICS.items():
            values = [float(record[record_key]) for record in model_records]
            row[f"{output_prefix}_mean"] = statistics.fmean(values)
            row[f"{output_prefix}_std"] = statistics.stdev(values)
        rows.append(row)
    return rows


def _format_number(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude < 1.0e-4 or magnitude >= 1.0e4:
        mantissa, exponent = f"{value:.4e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    return f"{value:.5g}"


def _format_mean_std(mean: float, std: float) -> str:
    return rf"{_format_number(mean)} \pm {_format_number(std)}"


def _latex_table(rows: Sequence[Mapping[str, Any]]) -> str:
    by_model = {str(row["model"]): row for row in rows}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{",
        "Behavior on held-out test patches containing no future fire activity.",
        "Lower values indicate better suppression of spurious background",
        "predictions.",
        "}",
        r"\label{tab:no_fire}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        "Model &",
        r"Mask FP $\downarrow$ &",
        r"Surf. Abs. Mean $\downarrow$ &",
        r"Canopy Abs. Mean $\downarrow$ &",
        r"Energy Abs. Mean $\downarrow$ \\",
        r"\midrule",
    ]
    for index, model in enumerate(MODELS):
        row = by_model[model]
        values = [
            _format_mean_std(float(row["mask_fp_mean"]), float(row["mask_fp_std"])),
            _format_mean_std(float(row["surface_abs_mean_mean"]), float(row["surface_abs_mean_std"])),
            _format_mean_std(float(row["canopy_abs_mean_mean"]), float(row["canopy_abs_mean_std"])),
            _format_mean_std(float(row["energy_abs_mean_mean"]), float(row["energy_abs_mean_std"])),
        ]
        lines.extend(
            [
                f"{DISPLAY_NAMES[model]} &",
                f"${values[0]}$ &",
                f"${values[1]}$ &",
                f"${values[2]}$ &",
                f"${values[3]}$ " + r"\\",
            ]
        )
        if index == 0:
            lines.append("")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _summary_text(records: Sequence[Mapping[str, Any]], subset_hash: str, output_root: Path) -> str:
    first = records[0]
    discrepancy = max(
        abs(float(record[key]))
        for record in records
        for key in (
            "surface_abs_mean_minus_mae",
            "canopy_abs_mean_minus_mae",
            "energy_log_abs_mean_minus_mae",
        )
    )
    lines = ["NO-FIRE TEST EVALUATION", ""]
    for model, label in (("baseline", "Baseline"), ("GA_Q2", "GA_Q2")):
        lines.append(f"{label} checkpoints:")
        for record in sorted((row for row in records if row["model"] == model), key=lambda row: int(row["seed"])):
            lines.append(f"  seed {record['seed']}: {record['checkpoint_path']}")
        lines.append("")
    lines.extend(
        [
            "Full test samples:",
            f"  {first['total_test_patches']}",
            "",
            "No-fire test samples:",
            f"  {first['no_fire_test_patches']}",
            "",
            "No-fire percentage:",
            f"  {float(first['no_fire_percentage']):.6g}%",
            "",
            "No-fire subset identical across models/seeds:",
            "  YES",
            "",
            "No-fire subset SHA-256:",
            f"  {subset_hash}",
            "",
            "Mask FP definition:",
            "  pixel-level sigmoid(mask_logit) > 0.5 false-positive rate",
            "",
            "Surface metric:",
            "  mean(abs(pred_surface))",
            "",
            "Canopy metric:",
            "  mean(abs(pred_canopy))",
            "",
            "Energy metric:",
            "  mean(abs(pred_energy_log))",
            "",
            "Absolute-prediction versus no-fire MAE maximum discrepancy:",
            f"  {discrepancy:.12g}",
            "",
            "Output:",
            f"  {output_root / 'no_fire_table.tex'}",
            "",
            "Submission command:",
            "  bash scripts/submit_no_fire_test.sh",
            "",
            "Aggregation command:",
            "  python scripts/evaluate_no_fire_test.py --aggregate-only",
            "",
            "NO RETRAINING PERFORMED:",
            "  YES",
        ]
    )
    return "\n".join(lines) + "\n"


def aggregate(
    *,
    cawfe_root: Path = CAWFE_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    per_fire_rows: list[dict[str, Any]] = []
    id_sets: list[dict[str, Any]] = []
    for model in MODELS:
        for seed in SEEDS:
            run_dir = discover_frozen_run(model, seed, cawfe_root)
            exact, reason = _artifact_has_exact_metrics(run_dir)
            if not exact:
                raise RuntimeError(
                    f"Aggregation is read-only and requires an exact completed artifact for {model} seed={seed}: {reason}. "
                    "Run the corresponding evaluation job first."
                )
            record, per_fire, all_ids, no_fire_ids = extract_record(model, seed, run_dir)
            records.append(record)
            per_fire_rows.extend(per_fire)
            id_sets.append(
                {
                    "model": model,
                    "seed": seed,
                    "full_hash": record["full_test_sample_ids_hash"],
                    "no_fire_hash": record["no_fire_subset_hash"],
                    "all_ids": sorted(all_ids),
                    "no_fire_ids": sorted(no_fire_ids),
                }
            )

    full_hashes = {str(item["full_hash"]) for item in id_sets}
    no_fire_hashes = {str(item["no_fire_hash"]) for item in id_sets}
    if len(full_hashes) != 1 or len(no_fire_hashes) != 1:
        raise RuntimeError("Full-test or no-fire sample IDs differ across model/seed evaluations")
    totals = {int(record["total_test_patches"]) for record in records}
    no_fire_counts = {int(record["no_fire_test_patches"]) for record in records}
    if len(totals) != 1 or len(no_fire_counts) != 1:
        raise RuntimeError("Test or no-fire patch counts differ across evaluations")

    output_root.mkdir(parents=True, exist_ok=True)
    by_seed_fields = (
        "model", "display_name", "seed", "checkpoint_path", "checkpoint_epoch", "run_dir",
        "test_metrics_path", "test_sample_metrics_path", "total_test_patches", "fire_test_patches",
        "no_fire_test_patches", "no_fire_percentage", "no_fire_pixel_count", "mask_fp",
        "surface_abs_mean", "canopy_abs_mean", "energy_log_abs_mean", "mask_probability_mean",
        "patch_fp_rate", "surface_signed_mean", "canopy_signed_mean", "energy_log_signed_mean",
        "surface_mae", "canopy_mae", "energy_log_mae", "surface_abs_mean_minus_mae",
        "canopy_abs_mean_minus_mae", "energy_log_abs_mean_minus_mae", "surface_rmse",
        "canopy_rmse", "energy_log_rmse", "surface_target_abs_mean", "canopy_target_abs_mean",
        "energy_log_target_abs_mean", "full_test_sample_ids_hash", "no_fire_subset_hash",
        "metric_scope", "test_used_for_model_selection",
    )
    _atomic_csv(output_root / "no_fire_test_by_seed.csv", records, by_seed_fields)
    _atomic_csv(
        output_root / "no_fire_test_per_fire.csv",
        per_fire_rows,
        (
            "model", "display_name", "seed", "fire_name", "no_fire_patch_count", "no_fire_pixel_count",
            "mask_fp", "surface_abs_mean", "canopy_abs_mean", "energy_log_abs_mean",
            "mask_probability_mean", "patch_fp_rate",
        ),
    )
    mean_std_rows = _aggregate_rows(records)
    _atomic_csv(
        output_root / "no_fire_test_mean_std.csv",
        mean_std_rows,
        (
            "model", "display_name", "seeds", "mask_fp_mean", "mask_fp_std",
            "surface_abs_mean_mean", "surface_abs_mean_std", "canopy_abs_mean_mean",
            "canopy_abs_mean_std", "energy_abs_mean_mean", "energy_abs_mean_std",
        ),
    )
    canonical = id_sets[0]
    sample_id_payload = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "split": "test",
        "membership_source": "ground_truth_future_target_mask_only",
        "target_mask_channel": FIRE_MASK_CHANNEL,
        "target_fire_pixel_rule": f"target_mask > {FIRE_MASK_THRESHOLD}",
        "active_fraction_threshold": ACTIVE_FRACTION_THRESHOLD,
        "prediction_threshold_not_used_for_membership": True,
        "full_test_sample_count": len(canonical["all_ids"]),
        "no_fire_sample_count": len(canonical["no_fire_ids"]),
        "hash_algorithm": "sha256_of_sorted_utf8_ids_joined_by_newline",
        "full_test_subset_hash": canonical["full_hash"],
        "no_fire_subset_hash": canonical["no_fire_hash"],
        "sample_ids": canonical["no_fire_ids"],
        "evaluations": [
            {
                "model": item["model"],
                "seed": item["seed"],
                "full_test_subset_hash": item["full_hash"],
                "no_fire_subset_hash": item["no_fire_hash"],
            }
            for item in id_sets
        ],
    }
    _atomic_json(output_root / "no_fire_test_sample_ids.json", sample_id_payload)
    _atomic_text(output_root / "no_fire_table.tex", _latex_table(mean_std_rows))
    summary = _summary_text(records, str(canonical["no_fire_hash"]), output_root)
    _atomic_text(output_root / "no_fire_summary.txt", summary)
    manifest = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "complete": True,
        "models": list(MODELS),
        "display_mapping": DISPLAY_NAMES,
        "seeds": list(SEEDS),
        "test_based_model_selection": False,
        "no_retraining_performed": True,
        "test_split": "complete_locked_held_out_test",
        "no_fire_membership": {
            "ground_truth_only": True,
            "mask_channel": FIRE_MASK_CHANNEL,
            "fire_mask_threshold": FIRE_MASK_THRESHOLD,
            "active_fraction_threshold": ACTIVE_FRACTION_THRESHOLD,
        },
        "mask_fp_definition": f"pixel-level sigmoid(mask_logit) > {PREDICTED_FIRE_THRESHOLD}",
        "continuous_metrics": {
            "surface": "mean(abs(prediction_channel_0))",
            "canopy": "mean(abs(prediction_channel_1))",
            "energy": "mean(abs(prediction_channel_3_log1p_energy))",
        },
        "full_test_sample_count": next(iter(totals)),
        "no_fire_sample_count": next(iter(no_fire_counts)),
        "full_test_subset_hash": canonical["full_hash"],
        "no_fire_subset_hash": canonical["no_fire_hash"],
        "outputs": {
            "by_seed": str(output_root / "no_fire_test_by_seed.csv"),
            "mean_std": str(output_root / "no_fire_test_mean_std.csv"),
            "per_fire": str(output_root / "no_fire_test_per_fire.csv"),
            "sample_ids": str(output_root / "no_fire_test_sample_ids.json"),
            "latex": str(output_root / "no_fire_table.tex"),
            "summary": str(output_root / "no_fire_summary.txt"),
        },
    }
    _atomic_json(output_root / "no_fire_analysis_manifest.json", manifest)
    print(summary, end="")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--aggregate-only", action="store_true", help="Read six completed artifacts; never load a model.")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate/reuse selected pairs without final aggregation.")
    parser.add_argument("--cawfe-root", type=Path, default=CAWFE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--list-seeds", action="store_true")
    args = parser.parse_args()
    if args.aggregate_only and args.evaluate_only:
        parser.error("--aggregate-only and --evaluate-only are mutually exclusive")
    return args


def main() -> None:
    args = _parse_args()
    if args.list_models:
        print("\n".join(MODELS))
        return
    if args.list_seeds:
        print("\n".join(str(seed) for seed in SEEDS))
        return
    if args.aggregate_only:
        aggregate(cawfe_root=args.cawfe_root, output_root=args.output_root)
        return
    evaluate_selection(
        args.models,
        args.seeds,
        cawfe_root=args.cawfe_root,
        output_root=args.output_root,
    )
    if not args.evaluate_only:
        aggregate(cawfe_root=args.cawfe_root, output_root=args.output_root)


if __name__ == "__main__":
    main()
