#!/usr/bin/env python3
"""Build plot-ready, multi-seed FLARE full-training data tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


DEFAULT_RUN_ROOT = Path("artifacts/final_training/cawfe_latte")
DEFAULT_ANALYSIS_ROOT = Path("artifacts/full_training/analysis")
DEFAULT_REGISTRY = Path("configs/final_training/cawfe_latte_finalists.yaml")
SAMPLING_PROTOCOL_ID = "epoch_random_subset_without_replacement_v1"
EXPECTED_BATCHES_PER_EPOCH = 7500
EXPECTED_BATCH_SIZE = 8

REQUIRED_RUN_FILES = (
    "metrics.json",
    "history/epoch_history.csv",
    "evaluation/per_fire_validation_metrics.csv",
    "evaluation/validation_sample_metrics.csv",
    "evaluation/qualitative_predictions.npz",
    "evaluation/efficiency_metrics.json",
    "metadata/training_sampling_protocol.json",
)
IDENTITY_FIELDS = ("model_name", "seed", "run_id", "run_dir")
NON_METRIC_EPOCH_FIELDS = {
    *IDENTITY_FIELDS,
    "architecture_name",
    "epoch",
    "global_step",
    "equivalent_full_dataset_epochs",
    "checkpoint_selection_metric_name",
    "validation_mode",
    "validation_scope",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    if not columns:
        raise ValueError(f"Cannot write CSV without fields: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "number_of_seeds": 0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "min": min(values),
        "max": max(values),
        "number_of_seeds": len(values),
    }


def _load_registry(path: Path) -> tuple[list[str], list[int]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    finalists = payload.get("finalists")
    seeds = payload.get("seeds")
    if not isinstance(finalists, Mapping) or not isinstance(seeds, list):
        raise ValueError(f"Invalid finalist registry: {path}")
    return [str(name) for name in finalists], [int(seed) for seed in seeds]


def _sampling_protocol_status(run_dir: Path) -> tuple[bool, str]:
    path = run_dir / "metadata" / "training_sampling_protocol.json"
    if not path.is_file():
        return False, "legacy_full_split_training"
    try:
        payload = _read_json(path)
    except Exception:
        return False, "invalid_training_sampling_protocol"
    expected = {
        "protocol_id": SAMPLING_PROTOCOL_ID,
        "batch_size": EXPECTED_BATCH_SIZE,
        "batches_per_epoch": EXPECTED_BATCHES_PER_EPOCH,
        "sampling_within_epoch": "without_replacement",
        "subset_changes_each_epoch": True,
    }
    if not isinstance(payload, Mapping) or any(payload.get(key) != value for key, value in expected.items()):
        return False, "incompatible_training_sampling_protocol"
    return True, "current_epoch_random_subset_training"


def _latest_complete_run(seed_dir: Path) -> tuple[Path | None, list[dict[str, Any]]]:
    if not seed_dir.is_dir():
        return None, []
    candidates: list[tuple[int, Path]] = []
    excluded: list[dict[str, Any]] = []
    for metrics_path in seed_dir.glob("*/metrics.json"):
        run_dir = metrics_path.parent
        protocol_ok, status = _sampling_protocol_status(run_dir)
        if not protocol_ok:
            excluded.append(
                {
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir.resolve()),
                    "status": status,
                }
            )
            continue
        if all((run_dir / relative).is_file() for relative in REQUIRED_RUN_FILES):
            candidates.append((metrics_path.stat().st_mtime_ns, run_dir))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1], excluded


def discover_runs(
    run_root: Path,
    registry_path: Path,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Choose the newest complete current-protocol run for every model/seed pair."""

    models, seeds = _load_registry(registry_path)
    runs: list[dict[str, Any]] = []
    missing: list[str] = []
    excluded: list[dict[str, Any]] = []
    for model_name in models:
        for seed in seeds:
            run_dir, seed_excluded = _latest_complete_run(run_root / model_name / f"seed_{seed}")
            excluded.extend(
                {"model_name": model_name, "seed": seed, **item}
                for item in seed_excluded
            )
            if run_dir is None:
                missing.append(f"{model_name} seed={seed}")
                continue
            runs.append(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "run_id": run_dir.name,
                    "run_dir": str(run_dir.resolve()),
                    "path": run_dir,
                }
            )
    return runs, missing, excluded


def _with_identity(rows: Iterable[Mapping[str, Any]], run: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        # Authoritative registry/path identity prevents stale embedded labels from
        # silently merging two seeds or architectures.
        for key in reversed(IDENTITY_FIELDS):
            row.pop(key, None)
        row = {key: run[key] for key in IDENTITY_FIELDS} | row
        output.append(row)
    return output


def _validate_qualitative_ids(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    reference: list[str] | None = None
    for run in runs:
        archive_path = Path(run["path"]) / "evaluation" / "qualitative_predictions.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            sample_ids = [str(value) for value in archive["sample_id"].tolist()]
        if len(sample_ids) != len(set(sample_ids)):
            raise RuntimeError(f"Duplicate qualitative sample IDs: {archive_path}")
        if reference is None:
            reference = sample_ids
        elif sample_ids != reference:
            raise RuntimeError(
                f"Qualitative sample IDs/order differ across runs: {archive_path}"
            )
    return reference or []


def _final_metric_rows(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        payload = _read_json(Path(run["path"]) / "metrics.json")
        full = payload.get("full_validation", {})
        efficiency = payload.get("efficiency", {})
        row: dict[str, Any] = {key: run[key] for key in IDENTITY_FIELDS}
        for key in ("best_epoch", "epochs_trained", "stopped_early", "source_ablation"):
            row[key] = payload.get(key)
        if isinstance(full, Mapping):
            row.update(full)
        if isinstance(efficiency, Mapping):
            for key, value in efficiency.items():
                if key not in row:
                    row[key] = value
                else:
                    row[f"efficiency_{key}"] = value
        rows.append(row)
    return rows


def _mean_std_by_model(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    models = sorted({str(row["model_name"]) for row in rows})
    excluded = {*IDENTITY_FIELDS, "best_epoch", "epochs_trained", "stopped_early", "source_ablation"}
    metric_names = sorted({key for row in rows for key in row if key not in excluded})
    for model_name in models:
        model_rows = [row for row in rows if str(row["model_name"]) == model_name]
        for metric in metric_names:
            values = [value for row in model_rows if (value := _finite(row.get(metric))) is not None]
            if not values:
                continue
            stopped_early_count = sum(
                str(row.get("stopped_early", "")).strip().lower() in {"1", "true", "yes"}
                for row in model_rows
            )
            output.append(
                {
                    "model_name": model_name,
                    "metric": metric,
                    **_stats(values),
                    "model_seed_count": len({int(row["seed"]) for row in model_rows}),
                    "stopped_early_count": stopped_early_count,
                }
            )
    return output


def _learning_curve_rows(epoch_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[tuple[int, float]]] = {}
    exposure_groups: dict[tuple[str, int], dict[int, float]] = {}
    step_groups: dict[tuple[str, int], dict[int, float]] = {}
    for row in epoch_rows:
        model_name = str(row["model_name"])
        epoch_number = int(float(row["epoch"]))
        seed = int(row["seed"])
        exposure = _finite(row.get("equivalent_full_dataset_epochs"))
        if exposure is not None:
            exposure_groups.setdefault((model_name, epoch_number), {})[seed] = exposure
        global_step = _finite(row.get("global_step"))
        if global_step is not None:
            step_groups.setdefault((model_name, epoch_number), {})[seed] = global_step
        for metric, raw_value in row.items():
            if metric in NON_METRIC_EPOCH_FIELDS:
                continue
            value = _finite(raw_value)
            if value is None:
                continue
            groups.setdefault((model_name, epoch_number, metric), []).append((seed, value))
    output: list[dict[str, Any]] = []
    for (model_name, epoch_number, metric), seed_values in sorted(groups.items()):
        by_seed = {seed: value for seed, value in seed_values}
        if len(by_seed) != len(seed_values):
            raise RuntimeError(f"Duplicate model/seed/epoch metric: {(model_name, epoch_number, metric)}")
        summary = _stats(list(by_seed.values()))
        exposure_values = list(exposure_groups.get((model_name, epoch_number), {}).values())
        step_values = list(step_groups.get((model_name, epoch_number), {}).values())
        exposure_summary = _stats(exposure_values)
        output.append(
            {
                "model_name": model_name,
                "epoch": epoch_number,
                "equivalent_full_dataset_epochs": exposure_summary["mean"],
                "equivalent_full_dataset_epochs_std": exposure_summary["std"],
                "mean_global_step": statistics.mean(step_values) if step_values else None,
                "metric": metric,
                "mean": summary["mean"],
                "std": summary["std"],
                "min": summary["min"],
                "max": summary["max"],
                "number_of_seeds": summary["number_of_seeds"],
                "n_seeds_at_epoch": summary["number_of_seeds"],
            }
        )
    return output


def summarize_full_training(
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
    registry_path: Path = DEFAULT_REGISTRY,
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
) -> dict[str, Any]:
    """Aggregate every available complete registered run into plot-ready tables."""

    runs, missing, excluded_legacy = discover_runs(run_root, registry_path)
    if not runs:
        raise RuntimeError(f"No complete full-training runs found under {run_root}")
    qualitative_ids = _validate_qualitative_ids(runs)

    epoch_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    per_fire_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for run in runs:
        run_path = Path(run["path"])
        run_epoch_rows = _with_identity(_read_csv(run_path / "history" / "epoch_history.csv"), run)
        epoch_numbers = [int(float(row["epoch"])) for row in run_epoch_rows]
        if len(epoch_numbers) != len(set(epoch_numbers)):
            raise RuntimeError(f"Duplicate epoch rows: {run_path}")
        epoch_rows.extend(run_epoch_rows)
        step_path = run_path / "history" / "step_history.csv"
        if step_path.is_file():
            step_rows.extend(_with_identity(_read_csv(step_path), run))
        per_fire_rows.extend(_with_identity(_read_csv(run_path / "evaluation" / "per_fire_validation_metrics.csv"), run))
        sample_rows.extend(_with_identity(_read_csv(run_path / "evaluation" / "validation_sample_metrics.csv"), run))
        efficiency_payload = _read_json(run_path / "evaluation" / "efficiency_metrics.json")
        efficiency_rows.extend(_with_identity([efficiency_payload], run))
        metrics_payload = _read_json(run_path / "metrics.json")
        best_rows.append(
            {
                **{key: run[key] for key in IDENTITY_FIELDS},
                "best_epoch": metrics_payload.get("best_epoch"),
                "epochs_trained": metrics_payload.get("epochs_trained"),
                "stopped_early": metrics_payload.get("stopped_early"),
                "best_screening_val_loss": metrics_payload.get("best_screening_validation", {}).get("total_loss"),
            }
        )

    final_rows = _final_metric_rows(runs)
    final_mean_std = _mean_std_by_model(final_rows)
    learning_rows = _learning_curve_rows(epoch_rows)
    analysis_root.mkdir(parents=True, exist_ok=True)
    _write_csv(analysis_root / "all_epoch_history.csv", epoch_rows, IDENTITY_FIELDS)
    _write_csv(analysis_root / "all_step_history.csv", step_rows, (*IDENTITY_FIELDS, "global_step", "epoch"))
    _write_csv(analysis_root / "final_metrics_by_run.csv", final_rows, IDENTITY_FIELDS)
    _write_csv(
        analysis_root / "final_metrics_mean_std.csv",
        final_mean_std,
        (
            "model_name", "metric", "mean", "std", "min", "max", "number_of_seeds",
            "model_seed_count", "stopped_early_count",
        ),
    )
    _write_csv(analysis_root / "per_fire_metrics_all_runs.csv", per_fire_rows, (*IDENTITY_FIELDS, "fire_name"))
    # The project environment intentionally has no Parquet engine. CSV is the
    # specification's dependency-free fallback and preserves every row/value.
    sample_output = analysis_root / "sample_metrics_all_runs.csv"
    _write_csv(sample_output, sample_rows, (*IDENTITY_FIELDS, "sample_id", "fire_name"))
    _write_csv(analysis_root / "efficiency_all_runs.csv", efficiency_rows, IDENTITY_FIELDS)
    _write_csv(analysis_root / "best_epochs.csv", best_rows, (*IDENTITY_FIELDS, "best_epoch"))
    _write_csv(
        analysis_root / "excluded_legacy_runs.csv",
        excluded_legacy,
        ("model_name", "seed", "run_id", "run_dir", "status"),
    )
    _write_csv(
        analysis_root / "learning_curves_mean_std.csv",
        learning_rows,
        (
            "model_name", "epoch", "equivalent_full_dataset_epochs",
            "equivalent_full_dataset_epochs_std", "mean_global_step", "metric",
            "mean", "std", "min", "max", "number_of_seeds", "n_seeds_at_epoch",
        ),
    )
    manifest = {
        "schema_version": 1,
        "run_root": str(run_root.resolve()),
        "analysis_root": str(analysis_root.resolve()),
        "complete_run_count": len(runs),
        "missing_registered_runs": missing,
        "excluded_legacy_runs": excluded_legacy,
        "required_sampling_protocol": {
            "protocol_id": SAMPLING_PROTOCOL_ID,
            "batch_size": EXPECTED_BATCH_SIZE,
            "batches_per_epoch": EXPECTED_BATCHES_PER_EPOCH,
        },
        "qualitative_sample_ids": qualitative_ids,
        "sample_metrics_format": "csv",
        "sample_metrics_path": str(sample_output.resolve()),
        "counts": {
            "epoch_rows": len(epoch_rows),
            "step_rows": len(step_rows),
            "per_fire_rows": len(per_fire_rows),
            "sample_rows": len(sample_rows),
            "efficiency_rows": len(efficiency_rows),
            "learning_curve_rows": len(learning_rows),
        },
    }
    (analysis_root / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    args = parser.parse_args()
    manifest = summarize_full_training(
        run_root=args.run_root,
        registry_path=args.registry,
        analysis_root=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
