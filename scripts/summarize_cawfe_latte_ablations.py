#!/usr/bin/env python3
"""Aggregate completed CAWFE-Latte screening runs without inventing an overall score."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path("artifacts/ablations/cawfe_latte")
REGISTRY_PATH = Path("configs/ablations/cawfe_latte_ablations.yaml")
METRIC_COLUMNS = {
    "Train Dice": ("train", "mask_dice"),
    "Screening Val Dice": ("validation", "mask_dice"),
    "Train IoU": ("train", "mask_iou"),
    "Screening Val IoU": ("validation", "mask_iou"),
    "Train Energy Log MAE": ("train", "energy_log_mae"),
    "Screening Val Energy Log MAE": ("validation", "energy_log_mae"),
    "Train Surface MAE": ("train", "surface_consumed_mae"),
    "Screening Val Surface MAE": ("validation", "surface_consumed_mae"),
    "Train Canopy MAE": ("train", "canopy_consumed_mae"),
    "Screening Val Canopy MAE": ("validation", "canopy_consumed_mae"),
    "Train Active Canopy MAE": ("train", "active_canopy_consumed_mae"),
    "Screening Val Active Canopy MAE": ("validation", "active_canopy_consumed_mae"),
    "Train Active Energy MAE": ("train", "active_energy_log_mae"),
    "Screening Val Active Energy MAE": ("validation", "active_energy_log_mae"),
    "Screening Val Fire Patch Count": ("validation", "fire_patch_count"),
    "Screening Val No-Fire Patch Count": ("validation", "no_fire_patch_count"),
    "Screening Val No-Fire Mask Prob Mean": ("validation", "no_fire_mask_prob_mean"),
    "Screening Val No-Fire Pixel FP Rate": ("validation", "no_fire_mask_false_positive_rate"),
    "Screening Val No-Fire Patch FP Rate": ("validation", "no_fire_patch_false_positive_rate"),
    "Screening Val No-Fire Surface Pred Mean": ("validation", "no_fire_surface_pred_mean"),
    "Screening Val No-Fire Canopy Pred Mean": ("validation", "no_fire_canopy_pred_mean"),
    "Screening Val No-Fire Energy Log Pred Mean": ("validation", "no_fire_energy_log_pred_mean"),
    "Patch Fire Accuracy": ("validation", "patch_fire_accuracy"),
    "Patch Fire F1": ("validation", "patch_fire_f1"),
    "Domain Loss": ("train", "domain_loss"),
    "Domain Accuracy": ("train", "domain_accuracy"),
    "Domain Random Chance": ("train", "domain_random_chance"),
    "MMD Loss": ("train", "mmd_loss"),
    "MMD Valid Batch Fraction": ("train", "mmd_valid_batch_fraction"),
    "Mask Guidance Alpha": ("train", "mask_guidance_alpha"),
    "Guidance Mean": ("validation", "guidance_mean"),
    "Guidance Std": ("validation", "guidance_std"),
    "Guidance Mean Active": ("validation", "guidance_mean_active"),
    "Guidance Mean Inactive": ("validation", "guidance_mean_inactive"),
    "Router Mean Expert 1": ("train", "expert_1_mean_weight"),
    "Router Mean Expert 2": ("train", "expert_2_mean_weight"),
    "Router Mean Expert 3": ("train", "expert_3_mean_weight"),
    "Router Entropy": ("train", "router_entropy"),
    "Router Max Probability Mean": ("train", "router_max_probability_mean"),
    "Load Balance Loss": ("train", "load_balance_loss"),
    "Contrastive Loss": ("train", "contrastive_loss"),
    "Contrastive Valid Anchor Fraction": ("train", "valid_contrastive_anchor_fraction"),
    "Batch Fraction No Fire": ("train", "batch_fraction_no_fire"),
    "Batch Fraction Tiny": ("train", "batch_fraction_tiny"),
    "Batch Fraction Small": ("train", "batch_fraction_small"),
    "Batch Fraction Medium": ("train", "batch_fraction_medium"),
    "Batch Fraction Large": ("train", "batch_fraction_large"),
    "Active-Fraction Aux MAE": ("validation", "physical_active_fraction_mae"),
    "Canopy-State Aux MAE": ("validation", "physical_canopy_state_mae"),
    "Energy-State Aux MAE": ("validation", "physical_energy_state_mae"),
}
FULL_VALIDATION_COLUMNS = {
    "Full Val Dice": "full_val_dice",
    "Full Val IoU": "full_val_iou",
    "Full Val Energy Log MAE": "full_val_energy_log_mae",
    "Full Val Surface MAE": "full_val_surface_mae",
    "Full Val Canopy MAE": "full_val_canopy_mae",
    "Full Val Active Canopy MAE": "full_val_active_canopy_mae",
}
FULL_VALIDATION_NO_FIRE_COLUMNS = {
    "Full Val Total Patches": "full_val_total_patch_count",
    "Full Val Fire Patches": "full_val_fire_patch_count",
    "Full Val No-Fire Patches": "full_val_no_fire_patch_count",
    "Full Val No-Fire %": "full_val_no_fire_patch_percent",
    "Full Val No-Fire Mask Prob Mean": "full_val_no_fire_mask_prob_mean",
    "Full Val No-Fire Pixel FP Rate": "full_val_no_fire_mask_false_positive_rate",
    "Full Val No-Fire Patch FP Rate": "full_val_no_fire_patch_false_positive_rate",
    "Full Val No-Fire Surface Pred Mean": "full_val_no_fire_surface_pred_mean",
    "Full Val No-Fire Canopy Pred Mean": "full_val_no_fire_canopy_pred_mean",
    "Full Val No-Fire Energy Log Pred Mean": "full_val_no_fire_energy_log_pred_mean",
    "Full Val No-Fire Energy MW Pred Mean": "full_val_no_fire_energy_mw_pred_mean",
}


DELTA_COLUMNS = {
    "Delta Full Val Dice": "Full Val Dice",
    "Delta Full Val IoU": "Full Val IoU",
    "Delta Full Val Energy Log MAE": "Full Val Energy Log MAE",
    "Delta Full Val Surface MAE": "Full Val Surface MAE",
    "Delta Full Val Canopy MAE": "Full Val Canopy MAE",
    "Delta Full Val Active Canopy MAE": "Full Val Active Canopy MAE",
}
PRIMARY_METRICS = {
    "Full Val Dice": True,
    "Full Val IoU": True,
    "Full Val Energy Log MAE": False,
    "Full Val Surface MAE": False,
    "Full Val Canopy MAE": False,
    "Full Val Active Canopy MAE": False,
}
PARENT_METRICS = {**PRIMARY_METRICS, "Full Val No-Fire Pixel FP Rate": False}
LEGACY_FULL_COLUMN_FALLBACKS = {
    "Full Val Dice": "Val Dice",
    "Full Val IoU": "Val IoU",
    "Full Val Energy Log MAE": "Val Energy Log MAE",
    "Full Val Surface MAE": "Val Surface MAE",
    "Full Val Canopy MAE": "Val Canopy MAE",
    "Full Val Active Canopy MAE": "Val Active Canopy MAE",
    "Full Val No-Fire Pixel FP Rate": "No-Fire Pixel FP Rate",
    "Full Val No-Fire Patch FP Rate": "No-Fire Patch FP Rate",
    "Full Val No-Fire Energy Log Pred Mean": "No-Fire Energy Log Pred Mean",
    "Full Val No-Fire Canopy Pred Mean": "No-Fire Canopy Pred Mean",
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
    "CGP": "CGP_separate_decoder_temporal_attention_mamba",
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
    "GA_Q1_fire_domain_adversarial": ["GA"],
    "GA_Q2_fire_domain_mmd": ["GA"],
    "GA_R_mask_guided_regression_attention": ["GA"],
    "GA_S_regression_moe": ["GA"],
    "GA_T_supervised_contrastive": ["GA"],
    "GA_U_physical_state_aux": ["GA"],
    "GPK_mamba_no_terrain": ["GP", "GK"],
    "CGPK_temporal_mamba_no_terrain": ["CGP", "GK"],
    "GAK_resblocks_no_terrain": ["GA", "GK"],
    "CGP_R_mask_guided_regression_attention": ["CGP"],
    "GP_R_mask_guided_regression_attention": ["GP"],
    "GK_R_mask_guided_regression_attention": ["GK"],
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
    for metric in PARENT_METRICS
]
BASE_COLUMNS = [
    "Ablation", "Components", "Change", "Parameters", "Train Time / Epoch",
    "Peak GPU Memory", "Best Epoch", *FULL_VALIDATION_COLUMNS, *FULL_VALIDATION_NO_FIRE_COLUMNS, *METRIC_COLUMNS, *DELTA_COLUMNS, "Run directory",
]
LEGACY_DELTA_COLUMNS = [name.replace("Delta Full Val ", "Delta Val ") for name in DELTA_COLUMNS]
LEGACY_PARENT_DELTA_COLUMNS = [
    f"Delta vs {reference} {LEGACY_FULL_COLUMN_FALLBACKS[metric]}"
    for reference in PARENT_REFERENCES
    for metric in PARENT_METRICS
]
COLUMNS = [
    *BASE_COLUMNS[:-1],
    *PARENT_DELTA_COLUMNS,
    *LEGACY_DELTA_COLUMNS,
    *LEGACY_PARENT_DELTA_COLUMNS,
    "Run directory",
]


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
        if not isinstance(metrics, Mapping):
            metrics = {}
        screening = payload.get("best_screening_validation")
        if isinstance(screening, Mapping):
            metrics = {**metrics, "validation": dict(screening)}
        training_section = payload.get("training")
        if isinstance(training_section, Mapping) and isinstance(training_section.get("best_epoch_metrics"), Mapping):
            metrics = {**metrics, "train": dict(training_section["best_epoch_metrics"])}
        history_row: dict[str, Any] = {}
        history_rows: list[dict[str, Any]] = []
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
        final_metrics = payload.get("final_epoch_metrics", {})
        final_history_row = history_rows[-1] if history_path.is_file() and history_rows else {}
        for column, (split, metric) in METRIC_COLUMNS.items():
            if metric == "mask_guidance_alpha":
                value = final_metrics.get(split, {}).get(metric)
                if value is None:
                    prefix = "train" if split == "train" else "val"
                    value = final_history_row.get(f"{prefix}_{metric}")
            else:
                value = metrics.get(split, {}).get(metric)
            if value is None:
                prefix = "train" if split == "train" else "val"
                value = history_row.get(f"{prefix}_{metric}")
                if value is None and metric == "fire_patch_count":
                    value = history_row.get(f"{prefix}_active_patch_count")
            row[column] = finite_or_none(value)
        full_metrics: dict[str, Any] = {}
        full_path = metrics_path.parent / "full_validation_metrics.json"
        if full_path.is_file():
            full_payload = json.loads(full_path.read_text(encoding="utf-8"))
            nested = full_payload.get("metrics", full_payload)
            if isinstance(nested, Mapping):
                full_metrics.update(nested)
        elif isinstance(payload.get("full_validation"), Mapping):
            full_metrics.update(payload["full_validation"])

        # Backward-compatible recovery only: old runs may have a no-fire-only
        # sidecar, but new runs must use full_validation_metrics.json.
        sidecar_path = metrics_path.parent / "no_fire_metrics.json"
        if sidecar_path.is_file():
            sidecar_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            nested_metrics = sidecar_payload.get("metrics", sidecar_payload)
            if isinstance(nested_metrics, Mapping):
                for key, value in nested_metrics.items():
                    full_metrics.setdefault(str(key), value)
        if "full_val_no_fire_patch_percent" not in full_metrics and "full_val_no_fire_percent" in full_metrics:
            full_metrics["full_val_no_fire_patch_percent"] = full_metrics["full_val_no_fire_percent"]
        for column, metric in {**FULL_VALIDATION_COLUMNS, **FULL_VALIDATION_NO_FIRE_COLUMNS}.items():
            row[column] = finite_or_none(full_metrics.get(metric))
        for full_column, legacy_column in LEGACY_FULL_COLUMN_FALLBACKS.items():
            if full_column in row:
                row[legacy_column] = row[full_column]
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


def _row_metric(row: Mapping[str, Any] | None, metric: str) -> Any:
    if row is None:
        return None
    value = row.get(metric)
    if value is not None:
        return value
    legacy = LEGACY_FULL_COLUMN_FALLBACKS.get(metric)
    return row.get(legacy) if legacy is not None else None


def add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    baseline = latest_rows_by_ablation(rows).get("baseline")
    for row in rows:
        for delta_column, metric_column in DELTA_COLUMNS.items():
            value = _row_metric(row, metric_column)
            reference = _row_metric(baseline, metric_column)
            delta = None if value is None or reference is None else float(value) - float(reference)
            row[delta_column] = delta
            row[delta_column.replace("Delta Full Val ", "Delta Val ")] = delta


def add_parent_deltas(rows: list[dict[str, Any]]) -> None:
    latest = latest_rows_by_ablation(rows)
    for row in rows:
        parent_deltas: dict[str, dict[str, float | None]] = {}
        for reference in COMBINATION_REFERENCES.get(str(row["Ablation"]), []):
            reference_row = latest.get(REFERENCE_NAMES[reference])
            metric_deltas: dict[str, float | None] = {}
            for metric in PARENT_METRICS:
                value = _row_metric(row, metric)
                reference_value = _row_metric(reference_row, metric)
                delta = None if value is None or reference_value is None else float(value) - float(reference_value)
                metric_deltas[metric] = delta
                legacy_metric = LEGACY_FULL_COLUMN_FALLBACKS[metric]
                metric_deltas[legacy_metric] = delta
                if reference != "baseline":
                    row[f"Delta vs {reference} {metric}"] = delta
                    row[f"Delta vs {reference} {legacy_metric}"] = delta
            parent_deltas[reference] = metric_deltas
        row["_parent_deltas"] = parent_deltas


def display(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def best_line(rows: list[dict[str, Any]], label: str, column: str, maximize: bool) -> str:
    candidates = [row for row in rows if isinstance(_row_metric(row, column), (int, float))]
    if not candidates:
        return f"{label}: N/A"
    selected = (max if maximize else min)(candidates, key=lambda row: float(_row_metric(row, column)))
    return f"{label}: {selected['Ablation']} ({display(_row_metric(selected, column))})"


def best_metric_lines(rows: list[dict[str, Any]], prefix: str = "Best") -> list[str]:
    return [
        best_line(rows, f"{prefix} Full Val Dice", "Full Val Dice", True),
        best_line(rows, f"{prefix} Full Val IoU", "Full Val IoU", True),
        best_line(rows, f"{prefix} Full Val Energy", "Full Val Energy Log MAE", False),
        best_line(rows, f"{prefix} Full Val Surface", "Full Val Surface MAE", False),
        best_line(rows, f"{prefix} Full Val Canopy", "Full Val Canopy MAE", False),
        best_line(rows, f"{prefix} Full Val Active Canopy", "Full Val Active Canopy MAE", False),
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
        status = comparison_status(_row_metric(row, metric), _row_metric(reference, metric), maximize)
        summary[status].append(metric.removeprefix("Full Val "))
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
    for name, row in latest.items():
        if name == "baseline":
            continue
        statuses = [comparison_status(_row_metric(row, metric), _row_metric(baseline, metric), maximize) for metric, maximize in PRIMARY_METRICS.items()]
        if len(statuses) == len(PRIMARY_METRICS) and all(status == "improved" for status in statuses):
            improved.append(str(row.get("_short_name", name)))
    return sorted(improved)


def pareto_candidates(rows: list[dict[str, Any]]) -> list[str]:
    """Return latest runs not dominated on their available primary metrics."""
    latest_rows = list(latest_rows_by_ablation(rows).values())
    candidates: list[str] = []
    for candidate in latest_rows:
        candidate_metrics = {
            metric for metric in PRIMARY_METRICS
            if isinstance(_row_metric(candidate, metric), (int, float))
        }
        if not candidate_metrics:
            continue
        dominated = False
        for challenger in latest_rows:
            if challenger is candidate:
                continue
            challenger_metrics = {
                metric for metric in PRIMARY_METRICS
                if isinstance(_row_metric(challenger, metric), (int, float))
            }
            # Missing candidate metrics are ignored. A challenger must cover every
            # metric the candidate does report; no value is ever fabricated.
            if not candidate_metrics.issubset(challenger_metrics):
                continue
            statuses = [
                comparison_status(_row_metric(challenger, metric), _row_metric(candidate, metric), PRIMARY_METRICS[metric])
                for metric in candidate_metrics
            ]
            if all(status in {"improved", "tied"} for status in statuses) and any(status == "improved" for status in statuses):
                dominated = True
                break
        if not dominated:
            candidates.append(str(candidate.get("_short_name", candidate["Ablation"])))
    return sorted(candidates)


def text_footer(rows: list[dict[str, Any]]) -> list[str]:
    separator = "=" * 60
    individuals = [row for row in rows if row["Ablation"] != "baseline" and len(row.get("_components", [])) <= 1]
    combinations = [row for row in rows if len(row.get("_components", [])) >= 2]
    improvers = all_metric_improvers(rows)
    pareto = pareto_candidates(rows)
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
        "BEST FULL-VALIDATION RESULTS",
        separator,
        *best_metric_lines(rows, "Best"),
        best_line(rows, "Lowest No-Fire Pixel FP", "Full Val No-Fire Pixel FP Rate", False),
        best_line(rows, "Lowest No-Fire Patch FP", "Full Val No-Fire Patch FP Rate", False),
        best_line(rows, "Lowest No-Fire Energy Prediction", "Full Val No-Fire Energy Log Pred Mean", False),
        best_line(rows, "Lowest No-Fire Canopy Prediction", "Full Val No-Fire Canopy Pred Mean", False),
        "",
        "MODELS IMPROVING ALL PRIMARY FULL-VALIDATION METRICS VS BASELINE",
        ", ".join(improvers) if improvers else "None",
        "",
        "PARETO / NON-DOMINATED CANDIDATES",
        ", ".join(pareto) if pareto else "None",
    ]


def markdown_footer(rows: list[dict[str, Any]]) -> list[str]:
    individuals = [row for row in rows if row["Ablation"] != "baseline" and len(row.get("_components", [])) <= 1]
    combinations = [row for row in rows if len(row.get("_components", [])) >= 2]
    improvers = all_metric_improvers(rows)
    pareto = pareto_candidates(rows)
    return [
        "## Best Individual Ablations", "", *best_metric_lines(individuals, "Best individual"), "",
        "## Best Combinations", "", *best_metric_lines(combinations, "Best combination"), "",
        "## Best Full-Validation Results", "", *best_metric_lines(rows, "Best"),
        best_line(rows, "Lowest No-Fire Pixel FP", "Full Val No-Fire Pixel FP Rate", False),
        best_line(rows, "Lowest No-Fire Patch FP", "Full Val No-Fire Patch FP Rate", False),
        best_line(rows, "Lowest No-Fire Energy Prediction", "Full Val No-Fire Energy Log Pred Mean", False),
        best_line(rows, "Lowest No-Fire Canopy Prediction", "Full Val No-Fire Canopy Pred Mean", False), "",
        "## Models Improving All Primary Full-Validation Metrics vs Baseline", "",
        ", ".join(improvers) if improvers else "None", "",
        "## Pareto / Non-Dominated Candidates", "",
        ", ".join(pareto) if pareto else "None", "",
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
