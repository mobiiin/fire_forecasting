#!/usr/bin/env python3
"""Aggregate CAWFE-Latte finalist validation results across registered seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

import yaml


DEFAULT_ROOT = Path("artifacts/final_training/cawfe_latte")
DEFAULT_REGISTRY = Path("configs/final_training/cawfe_latte_finalists.yaml")

COLUMN_METRICS = {
    "Dice": "full_val_dice",
    "IoU": "full_val_iou",
    "Precision": "full_val_precision",
    "Recall": "full_val_recall",
    "Surface MAE": "full_val_surface_mae",
    "Surface RMSE": "full_val_surface_rmse",
    "Canopy MAE": "full_val_canopy_mae",
    "Canopy RMSE": "full_val_canopy_rmse",
    "Energy Log MAE": "full_val_energy_log_mae",
    "Energy Log RMSE": "full_val_energy_log_rmse",
    "Energy MW MAE": "full_val_energy_mw_mae",
    "Energy MW RMSE": "full_val_energy_mw_rmse",
    "Active Surface MAE": "full_val_active_surface_mae",
    "Active Surface RMSE": "full_val_active_surface_rmse",
    "Active Canopy MAE": "full_val_active_canopy_mae",
    "Active Canopy RMSE": "full_val_active_canopy_rmse",
    "Active Energy Log MAE": "full_val_active_energy_log_mae",
    "Active Energy Log RMSE": "full_val_active_energy_log_rmse",
    "No-Fire Mask Probability": "full_val_no_fire_mask_prob_mean",
    "No-Fire Pixel FP": "full_val_no_fire_mask_false_positive_rate",
    "No-Fire Patch FP": "full_val_no_fire_patch_false_positive_rate",
    "No-Fire Surface MAE": "full_val_no_fire_surface_mae",
    "No-Fire Canopy MAE": "full_val_no_fire_canopy_mae",
    "No-Fire Energy Log MAE": "full_val_no_fire_energy_log_mae",
    "Inference Samples/s": "full_val_samples_per_second",
}
EFFICIENCY_COLUMNS = {
    "Params": "total_parameter_count",
    "Trainable Params": "trainable_parameter_count",
    "Training Time": "total_training_time_seconds",
    "Peak GPU Memory": "peak_gpu_memory_gb",
    "Mean Epoch Time": "mean_epoch_time_seconds",
}
PRIMARY_PAIRED = {
    "Dice": ("full_val_dice", "higher"),
    "IoU": ("full_val_iou", "higher"),
    "Energy Log MAE": ("full_val_energy_log_mae", "lower"),
    "Surface MAE": ("full_val_surface_mae", "lower"),
    "Canopy MAE": ("full_val_canopy_mae", "lower"),
    "Active Canopy MAE": ("full_val_active_canopy_mae", "lower"),
}
COMPACT_METRICS = ["Dice", "IoU", "Energy Log MAE", "Surface MAE", "Canopy MAE", "Active Canopy MAE"]
NO_FIRE_METRICS = [
    "No-Fire Mask Probability", "No-Fire Pixel FP", "No-Fire Patch FP",
    "No-Fire Surface MAE", "No-Fire Canopy MAE", "No-Fire Energy Log MAE",
]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _latest_complete_run(seed_dir: Path) -> Path | None:
    candidates: list[tuple[float, Path]] = []
    if not seed_dir.is_dir():
        return None
    for metrics_path in seed_dir.glob("*/metrics.json"):
        run_dir = metrics_path.parent
        if not (run_dir / "full_validation_metrics.json").is_file():
            continue
        if not (run_dir / "checkpoints" / "best_model.pt").is_file():
            continue
        candidates.append((metrics_path.stat().st_mtime, run_dir))
    return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]


def discover_rows(root: Path, registry_path: Path = DEFAULT_REGISTRY) -> tuple[list[dict[str, Any]], list[str], list[str], list[int]]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    architectures = list((registry.get("finalists") or {}).keys())
    seeds = [int(seed) for seed in registry.get("seeds", [])]
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for architecture in architectures:
        for seed in seeds:
            run_dir = _latest_complete_run(root / architecture / f"seed_{seed}")
            if run_dir is None:
                missing.append(f"{architecture} seed={seed}")
                continue
            try:
                metrics_payload = _read_json(run_dir / "metrics.json")
                full_payload = _read_json(run_dir / "full_validation_metrics.json")
                full = full_payload.get("metrics", {})
                if not isinstance(full, Mapping):
                    raise ValueError("full_validation_metrics.json has no metrics mapping")
                efficiency = metrics_payload.get("efficiency", {})
                if not isinstance(efficiency, Mapping):
                    efficiency = {}
                row: dict[str, Any] = {
                    "Architecture": architecture,
                    "Seed": seed,
                    "Best Epoch": metrics_payload.get("best_epoch"),
                    "Run Directory": str(run_dir),
                }
                for column, key in COLUMN_METRICS.items():
                    row[column] = _finite(full.get(key))
                for column, key in EFFICIENCY_COLUMNS.items():
                    row[column] = _finite(efficiency.get(key, metrics_payload.get(key)))
                rows.append(row)
            except Exception as exc:
                missing.append(f"{architecture} seed={seed} ({exc})")
    return rows, missing, architectures, seeds


def aggregate(rows: list[Mapping[str, Any]], architectures: list[str]) -> dict[str, dict[str, Any]]:
    numeric_columns = list(COLUMN_METRICS) + list(EFFICIENCY_COLUMNS)
    summary: dict[str, dict[str, Any]] = {}
    for architecture in architectures:
        selected = [row for row in rows if row["Architecture"] == architecture]
        entry: dict[str, Any] = {"Architecture": architecture, "Seed Count": len(selected)}
        for column in numeric_columns:
            values = [value for row in selected if (value := _finite(row.get(column))) is not None]
            entry[f"{column} Mean"] = statistics.mean(values) if values else None
            entry[f"{column} Std"] = statistics.stdev(values) if len(values) >= 2 else None
        summary[architecture] = entry
    return summary


def paired_comparisons(rows: list[Mapping[str, Any]], architectures: list[str]) -> dict[str, Any]:
    lookup = {(str(row["Architecture"]), int(row["Seed"])): row for row in rows}
    baseline_seeds = {seed for architecture, seed in lookup if architecture == "baseline"}
    result: dict[str, Any] = {}
    for architecture in architectures:
        if architecture == "baseline":
            continue
        shared = sorted(baseline_seeds & {seed for candidate, seed in lookup if candidate == architecture})
        metric_payload: dict[str, Any] = {}
        for label, (_key, direction) in PRIMARY_PAIRED.items():
            deltas = []
            by_seed = []
            for seed in shared:
                baseline_value = _finite(lookup[("baseline", seed)].get(label))
                contender_value = _finite(lookup[(architecture, seed)].get(label))
                if baseline_value is None or contender_value is None:
                    continue
                delta = contender_value - baseline_value
                deltas.append(delta)
                by_seed.append({"seed": seed, "delta": delta})
            metric_payload[label] = {
                "direction": direction,
                "by_seed": by_seed,
                "mean_delta": statistics.mean(deltas) if deltas else None,
                "std_delta": statistics.stdev(deltas) if len(deltas) >= 2 else None,
            }
        result[architecture] = {"shared_seeds": shared, "metrics": metric_payload}
    return result


def pareto_finalists(summary: Mapping[str, Mapping[str, Any]]) -> list[str]:
    candidates: dict[str, list[float]] = {}
    for architecture, entry in summary.items():
        values = [_finite(entry.get(f"{label} Mean")) for label in COMPACT_METRICS]
        if all(value is not None for value in values):
            candidates[architecture] = [float(value) for value in values]
    directions = ["higher", "higher", "lower", "lower", "lower", "lower"]
    nondominated: list[str] = []
    for candidate, values in candidates.items():
        dominated = False
        for other, other_values in candidates.items():
            if other == candidate:
                continue
            no_worse = all(
                left >= right if direction == "higher" else left <= right
                for left, right, direction in zip(other_values, values, directions)
            )
            strictly_better = any(
                left > right if direction == "higher" else left < right
                for left, right, direction in zip(other_values, values, directions)
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            nondominated.append(candidate)
    return nondominated


def _category(summary: Mapping[str, Mapping[str, Any]], label: str, maximize: bool) -> list[str]:
    values = {
        architecture: value
        for architecture, entry in summary.items()
        if (value := _finite(entry.get(f"{label} Mean"))) is not None
    }
    if not values:
        return []
    best = (max if maximize else min)(values.values())
    return [architecture for architecture, value in values.items() if value == best]


def categories(summary: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    return {
        "BEST MEAN VALIDATION DICE": _category(summary, "Dice", True),
        "BEST MEAN VALIDATION IOU": _category(summary, "IoU", True),
        "LOWEST MEAN ENERGY LOG MAE": _category(summary, "Energy Log MAE", False),
        "LOWEST MEAN SURFACE MAE": _category(summary, "Surface MAE", False),
        "LOWEST MEAN CANOPY MAE": _category(summary, "Canopy MAE", False),
        "LOWEST MEAN ACTIVE CANOPY MAE": _category(summary, "Active Canopy MAE", False),
        "BEST NO-FIRE BEHAVIOR": _category(summary, "No-Fire Patch FP", False),
        "MOST PARAMETER-EFFICIENT": _category(summary, "Params", False),
        "FASTEST INFERENCE": _category(summary, "Inference Samples/s", True),
    }


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(entry: Mapping[str, Any], label: str) -> str:
    mean = _finite(entry.get(f"{label} Mean"))
    std = _finite(entry.get(f"{label} Std"))
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.6g}±N/A"
    return f"{mean:.6g}±{std:.3g}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def render_report(
    summary: Mapping[str, Mapping[str, Any]],
    paired: Mapping[str, Any],
    missing: list[str],
    pareto: list[str],
    category_results: Mapping[str, list[str]],
    *,
    markdown: bool,
) -> str:
    lines = ["# CAWFE-Latte Finalist Results" if markdown else "CAWFE-LATTE FINALIST RESULTS", ""]
    if missing:
        lines.append(f"Incomplete registered runs: {len(missing)}")
        lines.extend(f"- {item}" for item in missing)
        lines.append("")
    headers = ["Model", "Seeds", *COMPACT_METRICS]
    table_rows = [
        [architecture, str(entry["Seed Count"]), *(_mean_std(entry, label) for label in COMPACT_METRICS)]
        for architecture, entry in summary.items()
    ]
    if markdown:
        lines.extend(_markdown_table(headers, table_rows))
    else:
        lines.append(" | ".join(headers))
        lines.extend(" | ".join(row) for row in table_rows)
    lines.extend(["", "## No-fire statistics" if markdown else "NO-FIRE STATISTICS", ""])
    no_fire_headers = ["Model", *NO_FIRE_METRICS]
    no_fire_rows = [
        [architecture, *(_mean_std(entry, label) for label in NO_FIRE_METRICS)]
        for architecture, entry in summary.items()
    ]
    if markdown:
        lines.extend(_markdown_table(no_fire_headers, no_fire_rows))
    else:
        lines.append(" | ".join(no_fire_headers))
        lines.extend(" | ".join(row) for row in no_fire_rows)

    lines.extend(["", "## Paired-seed deltas vs baseline" if markdown else "PAIRED-SEED DELTAS VS BASELINE", ""])
    lines.append("Delta is contender - baseline. Positive improves Dice/IoU; negative improves MAE. No significance claims are made from three seeds.")
    for architecture, payload in paired.items():
        lines.extend(["", architecture])
        for label, metric in payload["metrics"].items():
            seeds = ", ".join(f"{item['seed']}:{item['delta']:.6g}" for item in metric["by_seed"]) or "N/A"
            mean = _finite(metric["mean_delta"])
            std = _finite(metric["std_delta"])
            mean_text = "N/A" if mean is None else f"{mean:.6g}"
            std_text = "N/A" if std is None else f"{std:.3g}"
            lines.append(f"- {label}: [{seeds}] mean={mean_text} std={std_text}")

    lines.extend(["", "## Category results" if markdown else "CATEGORY RESULTS", ""])
    for label, winners in category_results.items():
        lines.append(f"{label}: {', '.join(winners) if winners else 'N/A'}")
    lines.extend(["", "## Pareto / non-dominated finalists" if markdown else "PARETO / NON-DOMINATED FINALISTS", ""])
    lines.append(", ".join(pareto) if pareto else "N/A (insufficient complete metrics)")
    lines.append("")
    return "\n".join(lines)


def summarize(root: Path, registry_path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    rows, missing, architectures, seeds = discover_rows(root, registry_path)
    summary = aggregate(rows, architectures)
    paired = paired_comparisons(rows, architectures)
    pareto = pareto_finalists(summary)
    category_results = categories(summary)

    per_seed_fields = [
        "Architecture", "Seed", "Best Epoch", *COLUMN_METRICS.keys(), *EFFICIENCY_COLUMNS.keys(), "Run Directory",
    ]
    _write_csv(root / "finalist_results_by_seed.csv", rows, per_seed_fields)
    summary_rows = [summary[name] for name in architectures]
    summary_fields = ["Architecture", "Seed Count"] + [
        f"{column} {suffix}"
        for column in list(COLUMN_METRICS) + list(EFFICIENCY_COLUMNS)
        for suffix in ("Mean", "Std")
    ]
    _write_csv(root / "finalist_results_summary.csv", summary_rows, summary_fields)
    payload = {
        "schema_version": 1,
        "registered_architectures": architectures,
        "registered_seeds": seeds,
        "completed_run_count": len(rows),
        "expected_run_count": len(architectures) * len(seeds),
        "missing_or_incomplete_runs": missing,
        "by_seed": rows,
        "summary": summary,
        "paired_seed_comparisons_vs_baseline": paired,
        "categories": category_results,
        "pareto_non_dominated_finalists": pareto,
        "note": "No significance claims are made from three seeds; no weighted overall score is computed.",
    }
    (root / "finalist_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "finalist_results.md").write_text(
        render_report(summary, paired, missing, pareto, category_results, markdown=True), encoding="utf-8"
    )
    (root / "finalist_results.txt").write_text(
        render_report(summary, paired, missing, pareto, category_results, markdown=False), encoding="utf-8"
    )
    for item in missing:
        print(f"WARNING: incomplete or missing run: {item}", file=sys.stderr)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    payload = summarize(args.root, args.registry)
    print(f"Completed runs: {payload['completed_run_count']}/{payload['expected_run_count']}")
    print(args.root / "finalist_results.txt")


if __name__ == "__main__":
    main()
