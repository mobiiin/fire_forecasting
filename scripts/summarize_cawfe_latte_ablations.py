#!/usr/bin/env python3
"""Aggregate completed CAWFE-Latte screening runs without inventing an overall score."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("artifacts/ablations/cawfe_latte")
REGISTRY_PATH = Path("configs/ablations/cawfe_latte_ablations.yaml")
METRIC_COLUMNS = {
    "Train Dice": ("train", "mask_dice"),
    "Val Dice": ("validation", "mask_dice"),
    "Train IoU": ("train", "mask_iou"),
    "Val IoU": ("validation", "mask_iou"),
    "Train Energy Log MAE": ("train", "energy_log_mae"),
    "Val Energy Log MAE": ("validation", "energy_log_mae"),
    "Train Surface MAE": ("train", "surface_consumed_mae"),
    "Val Surface MAE": ("validation", "surface_consumed_mae"),
    "Train Canopy MAE": ("train", "canopy_consumed_mae"),
    "Val Canopy MAE": ("validation", "canopy_consumed_mae"),
    "Train Active Canopy MAE": ("train", "active_canopy_consumed_mae"),
    "Val Active Canopy MAE": ("validation", "active_canopy_consumed_mae"),
    "Train Active Energy MAE": ("train", "active_energy_log_mae"),
    "Val Active Energy MAE": ("validation", "active_energy_log_mae"),
    "Val No-Fire Patch Count": ("validation", "no_fire_patch_count"),
    "Val No-Fire Mask Prob Mean": ("validation", "no_fire_mask_prob_mean"),
    "No-Fire FP": ("validation", "no_fire_mask_false_positive_rate"),
    "Val No-Fire Surface Pred Mean": ("validation", "no_fire_surface_pred_mean"),
    "Val No-Fire Canopy Pred Mean": ("validation", "no_fire_canopy_pred_mean"),
    "Val No-Fire Energy Log Pred Mean": ("validation", "no_fire_energy_log_pred_mean"),
    "Patch Fire Accuracy": ("validation", "patch_fire_accuracy"),
    "Patch Fire F1": ("validation", "patch_fire_f1"),
}
DELTA_COLUMNS = {
    "Delta Val Dice": "Val Dice",
    "Delta Val IoU": "Val IoU",
    "Delta Val Energy Log MAE": "Val Energy Log MAE",
    "Delta Val Surface MAE": "Val Surface MAE",
    "Delta Val Canopy MAE": "Val Canopy MAE",
    "Delta Val Active Canopy MAE": "Val Active Canopy MAE",
}
PRIMARY_METRICS = {
    "Val Dice": True,
    "Val IoU": True,
    "Val Energy Log MAE": False,
    "Val Surface MAE": False,
    "Val Canopy MAE": False,
    "Val Active Canopy MAE": False,
}
REFERENCE_NAMES = {
    "baseline": "baseline",
    "A": "A_resblocks",
    "C": "C_temporal_attention",
    "E": "E_earthformer_lite",
    "G": "G_separate_regression_decoder",
    "K": "K_no_terrain_film",
    "P": "P_mamba_postfusion",
    "CG": "CG_separate_decoder_temporal_attention",
    "GA": "GA_separate_decoder_resblocks",
    "GE": "GE_separate_decoder_earthformer",
    "GP": "GP_separate_decoder_mamba",
    "GK": "GK_separate_decoder_no_terrain",
}
COMBINATION_REFERENCES = {
    "CG_separate_decoder_temporal_attention": ["baseline", "C", "G"],
    "GA_separate_decoder_resblocks": ["baseline", "A", "G"],
    "GE_separate_decoder_earthformer": ["baseline", "E", "G"],
    "GP_separate_decoder_mamba": ["baseline", "P", "G"],
    "GK_separate_decoder_no_terrain": ["baseline", "K", "G"],
    "CGA_separate_decoder_temporal_attention_resblocks": ["baseline", "CG", "GA", "A", "C", "G"],
    "CGE_separate_decoder_temporal_attention_earthformer": ["baseline", "CG", "GE", "E", "C", "G"],
    "CGP_separate_decoder_temporal_attention_mamba": ["baseline", "CG", "GP", "P", "C", "G"],
    "CGK_separate_decoder_temporal_attention_no_terrain": ["baseline", "CG", "GK", "K", "C", "G"],
}
INDIVIDUAL_COMPONENTS = {
    "A_resblocks": ["A"], "B_multiscale_context": ["B"], "C_temporal_attention": ["C"],
    "D_local_window_attention": ["D"], "E_earthformer_lite": ["E"],
    "G_separate_regression_decoder": ["G"], "I_patch_fire_classifier": ["I"],
    "K_no_terrain_film": ["K"], "L_simple_concat_fusion": ["L"],
    "O_fourier_postfusion": ["O"], "P_mamba_postfusion": ["P"],
}
PARENT_REFERENCES = []
for references in COMBINATION_REFERENCES.values():
    for reference in references:
        if reference != "baseline" and reference not in PARENT_REFERENCES:
            PARENT_REFERENCES.append(reference)
PARENT_DELTA_COLUMNS = [
    f"Delta vs {reference} {metric}"
    for reference in PARENT_REFERENCES
    for metric in PRIMARY_METRICS
]
BASE_COLUMNS = [
    "Ablation", "Components", "Change", "Parameters", "Train Time / Epoch",
    "Peak GPU Memory", "Best Epoch", *METRIC_COLUMNS, *DELTA_COLUMNS, "Run directory",
]
COLUMNS = [*BASE_COLUMNS[:-1], *PARENT_DELTA_COLUMNS, "Run directory"]


def finite_or_none(value: Any) -> int | float | str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if math.isfinite(number) else None


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(name): dict(entry) for name, entry in payload.get("ablations", {}).items()}


def discover_rows(root: Path, registry_path: Path = REGISTRY_PATH) -> list[dict[str, Any]]:
    registry = load_registry(registry_path)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*/*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload.get("best_epoch_metrics", {})
        history_row: dict[str, Any] = {}
        history_path = metrics_path.parent / "training_history.csv"
        if history_path.is_file():
            with history_path.open(encoding="utf-8", newline="") as handle:
                history_rows = list(csv.DictReader(handle))
            best_epoch = int(payload.get("best_epoch", -1))
            matches = [item for item in history_rows if int(float(item.get("epoch", -1))) == best_epoch]
            if matches:
                history_row = matches[-1]
        ablation = str(payload.get("ablation", metrics_path.parents[1].name))
        entry = registry.get(ablation, {})
        components = payload.get("components", entry.get("components", INDIVIDUAL_COMPONENTS.get(ablation, [])))
        if not isinstance(components, list):
            components = []
        row: dict[str, Any] = {
            "Ablation": ablation,
            "Components": ", ".join(str(value) for value in components) if components else "baseline",
            "Change": payload.get("exact_change", payload.get("changed_component")),
            "Parameters": payload.get("parameter_count"),
            "Train Time / Epoch": finite_or_none(payload.get("train_time_per_epoch_sec")),
            "Peak GPU Memory": finite_or_none(payload.get("peak_gpu_memory_gb")),
            "Best Epoch": payload.get("best_epoch"),
            "Run directory": str(metrics_path.parent),
            "_components": list(components),
            "_short_name": str(payload.get("short_name", entry.get("short_name", components[0] if len(components) == 1 else ablation))),
            "_mtime": metrics_path.stat().st_mtime,
        }
        for column, (split, metric) in METRIC_COLUMNS.items():
            value = metrics.get(split, {}).get(metric)
            if value is None:
                prefix = "train" if split == "train" else "val"
                value = history_row.get(f"{prefix}_{metric}")
            row[column] = finite_or_none(value)
        rows.append(row)
    rows.sort(key=lambda row: (str(row["Ablation"]), str(row["Run directory"])))
    return rows


def latest_rows_by_ablation(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row["Ablation"])
        if name not in latest or float(row["_mtime"]) > float(latest[name]["_mtime"]):
            latest[name] = row
    return latest


def add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    baseline = latest_rows_by_ablation(rows).get("baseline")
    for row in rows:
        for delta_column, metric_column in DELTA_COLUMNS.items():
            value = row.get(metric_column)
            reference = baseline.get(metric_column) if baseline else None
            row[delta_column] = None if value is None or reference is None else float(value) - float(reference)


def add_parent_deltas(rows: list[dict[str, Any]]) -> None:
    latest = latest_rows_by_ablation(rows)
    for row in rows:
        parent_deltas: dict[str, dict[str, float | None]] = {}
        for reference in COMBINATION_REFERENCES.get(str(row["Ablation"]), []):
            reference_row = latest.get(REFERENCE_NAMES[reference])
            metric_deltas: dict[str, float | None] = {}
            for metric in PRIMARY_METRICS:
                value = row.get(metric)
                reference_value = reference_row.get(metric) if reference_row else None
                delta = None if value is None or reference_value is None else float(value) - float(reference_value)
                metric_deltas[metric] = delta
                if reference != "baseline":
                    row[f"Delta vs {reference} {metric}"] = delta
            parent_deltas[reference] = metric_deltas
        row["_parent_deltas"] = parent_deltas


def display(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def best_line(rows: list[dict[str, Any]], label: str, column: str, maximize: bool) -> str:
    candidates = [row for row in rows if isinstance(row.get(column), (int, float))]
    if not candidates:
        return f"{label}: N/A"
    selected = (max if maximize else min)(candidates, key=lambda row: float(row[column]))
    return f"{label}: {selected['Ablation']} ({display(selected[column])})"


def best_metric_lines(rows: list[dict[str, Any]], prefix: str = "Best") -> list[str]:
    return [
        best_line(rows, f"{prefix} Val Dice", "Val Dice", True),
        best_line(rows, f"{prefix} Val IoU", "Val IoU", True),
        best_line(rows, f"{prefix} Energy", "Val Energy Log MAE", False),
        best_line(rows, f"{prefix} Surface", "Val Surface MAE", False),
        best_line(rows, f"{prefix} Canopy", "Val Canopy MAE", False),
        best_line(rows, f"{prefix} Active Canopy", "Val Active Canopy MAE", False),
    ]


def comparison_status(value: Any, reference: Any, maximize: bool) -> str:
    if not isinstance(value, (int, float)) or not isinstance(reference, (int, float)):
        return "unavailable"
    difference = float(value) - float(reference)
    if math.isclose(difference, 0.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
        return "tied"
    return "improved" if (difference > 0.0) == maximize else "regressed"


def comparison_summary(row: dict[str, Any], reference: dict[str, Any] | None) -> dict[str, list[str]]:
    summary = {status: [] for status in ("improved", "regressed", "tied", "unavailable")}
    for metric, maximize in PRIMARY_METRICS.items():
        status = comparison_status(row.get(metric), reference.get(metric) if reference else None, maximize)
        summary[status].append(metric.removeprefix("Val "))
    return summary


def combination_synergy_lines(rows: list[dict[str, Any]]) -> list[str]:
    latest = latest_rows_by_ablation(rows)
    lines = ["Combination Synergy"]
    for name, references in COMBINATION_REFERENCES.items():
        row = latest.get(name)
        short_name = next((key for key, value in REFERENCE_NAMES.items() if value == name), None) or name.split("_", 1)[0]
        if row is None:
            lines.append(f"{short_name}: no completed run")
            continue
        lines.append(f"{short_name}:")
        for reference in references:
            summary = comparison_summary(row, latest.get(REFERENCE_NAMES[reference]))
            available = len(PRIMARY_METRICS) - len(summary["unavailable"])
            all_improved = available > 0 and len(summary["improved"]) == available
            parts = [f"outperformed on all available primary metrics: {'YES' if all_improved else 'NO'}"]
            for status in ("improved", "regressed", "tied", "unavailable"):
                if summary[status]:
                    parts.append(f"{status}: {', '.join(summary[status])}")
            lines.append(f"  vs {reference}: " + "; ".join(parts))
    return lines


def all_metric_improvers(rows: list[dict[str, Any]]) -> list[str]:
    latest = latest_rows_by_ablation(rows)
    baseline = latest.get("baseline")
    if baseline is None:
        return []
    improved: list[str] = []
    for name in COMBINATION_REFERENCES:
        row = latest.get(name)
        if row is None:
            continue
        statuses = [comparison_status(row.get(metric), baseline.get(metric), maximize) for metric, maximize in PRIMARY_METRICS.items()]
        available = [status for status in statuses if status != "unavailable"]
        if available and all(status == "improved" for status in available):
            improved.append(str(row.get("_short_name", name)))
    return improved


def text_footer(rows: list[dict[str, Any]]) -> list[str]:
    separator = "=" * 60
    individuals = [row for row in rows if row["Ablation"] != "baseline" and len(row.get("_components", [])) <= 1]
    combinations = [row for row in rows if len(row.get("_components", [])) >= 2]
    improvers = all_metric_improvers(rows)
    return [
        separator,
        "BEST INDIVIDUAL ABLATIONS",
        separator,
        *best_metric_lines(individuals, "Best individual"),
        "",
        separator,
        "BEST COMBINATIONS",
        separator,
        *best_metric_lines(combinations, "Best combination"),
        "",
        separator,
        "BEST OVERALL ACROSS ALL RUNS",
        separator,
        *best_metric_lines(rows, "Best"),
        "",
        "Combination(s) that improve ALL available primary validation metrics relative to baseline: "
        + (", ".join(improvers) if improvers else "None"),
    ]


def markdown_footer(rows: list[dict[str, Any]]) -> list[str]:
    individuals = [row for row in rows if row["Ablation"] != "baseline" and len(row.get("_components", [])) <= 1]
    combinations = [row for row in rows if len(row.get("_components", [])) >= 2]
    improvers = all_metric_improvers(rows)
    return [
        "## Best Individual Ablations", "", *best_metric_lines(individuals, "Best individual"), "",
        "## Best Combinations", "", *best_metric_lines(combinations, "Best combination"), "",
        "## Best Overall Across All Runs", "", *best_metric_lines(rows, "Best"), "",
        "Combination(s) that improve ALL available primary validation metrics relative to baseline: "
        + (", ".join(improvers) if improvers else "None"), "",
    ]


def write_outputs(root: Path, rows: list[dict[str, Any]]) -> None:
    serializable = []
    for row in rows:
        payload = {column: row.get(column) for column in COLUMNS}
        payload["Parent Deltas"] = row.get("_parent_deltas", {})
        serializable.append(payload)
    csv_path = root / "ablation_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in COLUMNS} for row in rows)
    (root / "ablation_results.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    header = "| " + " | ".join(BASE_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in BASE_COLUMNS) + " |"
    markdown_rows = [
        "| " + " | ".join(display(row.get(column)) for column in BASE_COLUMNS) + " |"
        for row in rows
    ]
    synergy = combination_synergy_lines(rows)
    markdown_synergy = ["## Combination Synergy", ""] + [f"- {line}" if not line.endswith(":") else f"### {line}" for line in synergy[1:]]
    markdown = "\n".join(
        ["# CAWFE-Latte Ablation Results", "", header, separator, *markdown_rows, "", *markdown_footer(rows), *markdown_synergy, ""]
    )
    (root / "ablation_results.md").write_text(markdown, encoding="utf-8")

    widths = {
        column: max(len(column), *(len(display(row.get(column))) for row in rows))
        for column in BASE_COLUMNS
    }
    text_header = "  ".join(column.ljust(widths[column]) for column in BASE_COLUMNS)
    text_rows = [
        "  ".join(display(row.get(column)).ljust(widths[column]) for column in BASE_COLUMNS)
        for row in rows
    ]
    text_report = "\n".join([text_header, *text_rows, "", *text_footer(rows), "", *combination_synergy_lines(rows), ""])
    (root / "ablation_results.txt").write_text(text_report, encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = discover_rows(ROOT)
    add_baseline_deltas(rows)
    add_parent_deltas(rows)
    write_outputs(ROOT, rows)
    print(f"Summarized {len(rows)} completed run(s) under {ROOT}.")


if __name__ == "__main__":
    main()
