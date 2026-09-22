#!/usr/bin/env python3
"""Aggregate complete locked-test runs into the paper's Table 2 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import yaml

from scripts.run_table2_baseline import (
	_candidate_identity,
	_complete_test_artifact,
	_compatible_candidates,
	_has_best_checkpoint,
	_has_complete_validation,
	evaluate_existing_cawfe_run,
	output_parent,
)
from src.config import load_config
from src.evaluation.table2_protocol import identities_match, table2_protocol_identity


REGISTRY_PATH = Path("configs/table2_baselines/table2_baselines.yaml")
OUTPUT_ROOT = Path("artifacts/table2_baselines")
CAWFE_ROOT = Path("artifacts/final_training/cawfe_latte")
SUMMARY_ROOT = OUTPUT_ROOT / "summary"

ROW_ORDER = (
	"persistence",
	"linear_extrapolation",
	"convlstm_unet",
	"earthformer_lite",
	"cawfe_st_mamba",
	"cawfe_latte_baseline",
	"cawfe_latte_final",
)
METRIC_COLUMNS = {
	"Dice": "test_dice",
	"IoU": "test_iou",
	"Surface MAE": "test_surface_mae",
	"Canopy MAE": "test_canopy_mae",
	"Energy Log MAE": "test_energy_log_mae",
	"Active Canopy MAE": "test_active_canopy_mae",
}


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_suffix(path.suffix + ".tmp")
	temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	temporary.replace(path)


def _load_test_artifact(run_dir: Path) -> dict[str, Any]:
	path = run_dir / "evaluation" / "test_metrics.json"
	if not _complete_test_artifact(run_dir):
		raise RuntimeError(f"Incomplete or invalid held-out test artifact: {path}")
	payload = json.loads(path.read_text(encoding="utf-8"))
	if bool(payload.get("test_used_for_model_selection", True)):
		raise RuntimeError(f"Test artifact does not certify validation-only model selection: {path}")
	return payload


def _discover_external_runs(registry: Mapping[str, Any], output_root: Path) -> tuple[dict[str, list[Path]], list[dict[str, Any]]]:
	selected: dict[str, list[Path]] = {}
	missing: list[dict[str, Any]] = []
	seeds = [int(seed) for seed in registry["seeds"]]
	for baseline, entry in registry["baselines"].items():
		learned = bool(entry["learned"])
		config = load_config(entry["config_path"])
		if not learned:
			run_dir = output_root / baseline
			identity = table2_protocol_identity(config, baseline, None)
			if _complete_test_artifact(run_dir) and identities_match(identity, _candidate_identity(run_dir, baseline, None)):
				selected[baseline] = [run_dir]
			else:
				missing.append({"model": baseline, "seed": None, "reason": "missing_complete_compatible_test_run"})
			continue
		selected_runs: list[Path] = []
		for seed in seeds:
			training = dict(config.get("training", {}))
			training["seed"] = seed
			resolved = dict(config)
			resolved["training"] = training
			resolved["seed"] = seed
			identity = table2_protocol_identity(resolved, baseline, seed)
			parent = output_root / baseline / f"seed_{seed}"
			candidates = [
				run_dir
				for run_dir in _compatible_candidates(parent, baseline, seed, identity)
				if _has_best_checkpoint(run_dir) and _has_complete_validation(run_dir) and _complete_test_artifact(run_dir)
			]
			if candidates:
				selected_runs.append(candidates[0])
			else:
				missing.append({"model": baseline, "seed": seed, "reason": "missing_complete_compatible_test_run"})
		selected[baseline] = selected_runs
	return selected, missing


def _discover_cawfe_training_run(finalist: str, seed: int, root: Path) -> Path | None:
	parent = root / finalist / f"seed_{seed}"
	if not parent.is_dir():
		return None
	candidates: list[Path] = []
	for run_dir in parent.iterdir():
		if not run_dir.is_dir():
			continue
		if not (run_dir / "checkpoints" / "best_model.pt").is_file():
			continue
		if not (run_dir / "full_validation_metrics.json").is_file():
			continue
		config_path = run_dir / "resolved_config.yaml"
		if not config_path.is_file():
			config_path = run_dir / "configs" / "resolved_config.yaml"
		if not config_path.is_file():
			continue
		try:
			config = load_config(config_path)
			protocol = config.get("training", {}).get("sampling_protocol", {})
			if (
				str(config.get("final_training", {}).get("finalist")) == finalist
				and int(config.get("training", {}).get("seed", -1)) == seed
				and protocol.get("protocol_id") == "epoch_random_subset_without_replacement_v1"
				and int(protocol.get("batches_per_epoch", 0)) == 7500
				and int(protocol.get("batch_size", 0)) == 8
			):
				candidates.append(run_dir)
		except (KeyError, TypeError, ValueError):
			continue
	return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _ensure_cawfe_test_runs(
	registry: Mapping[str, Any],
	root: Path,
	*,
	evaluate_missing: bool,
) -> tuple[dict[str, list[Path]], list[dict[str, Any]]]:
	selected: dict[str, list[Path]] = {}
	missing: list[dict[str, Any]] = []
	seeds = [int(seed) for seed in registry["seeds"]]
	for table_name, entry in registry["cawfe_latte_rows"].items():
		finalist = str(entry["finalist"])
		runs: list[Path] = []
		for seed in seeds:
			run_dir = _discover_cawfe_training_run(finalist, seed, root)
			if run_dir is None:
				missing.append({"model": table_name, "seed": seed, "reason": "compatible_finalist_checkpoint_not_found"})
				continue
			if not _complete_test_artifact(run_dir):
				if evaluate_missing:
					print(f"Evaluating frozen {table_name} seed={seed} on locked test: {run_dir}")
					evaluate_existing_cawfe_run(run_dir, table_name, seed)
				else:
					missing.append({"model": table_name, "seed": seed, "reason": "test_metrics_missing"})
					continue
			runs.append(run_dir)
		selected[table_name] = runs
	return selected, missing


def _aggregate_model(model_key: str, display_name: str, run_dirs: Sequence[Path]) -> dict[str, Any]:
	artifacts = [_load_test_artifact(run_dir) for run_dir in run_dirs]
	seed_values = [artifact.get("seed") for artifact in artifacts]
	deterministic = all(seed is None for seed in seed_values)
	metrics: dict[str, Any] = {}
	for label, key in METRIC_COLUMNS.items():
		values = [float(artifact["metrics"][key]) for artifact in artifacts]
		metrics[label] = {
			"mean": statistics.fmean(values),
			"std": None if deterministic or len(values) == 1 else statistics.stdev(values),
			"values": values,
		}
	return {
		"model_key": model_key,
		"Model": display_name,
		"Seeds": "deterministic" if deterministic else ",".join(str(int(seed)) for seed in seed_values),
		"seed_values": seed_values,
		"run_dirs": [str(path) for path in run_dirs],
		"metrics": metrics,
	}


def _format_metric(metric: Mapping[str, Any]) -> str:
	mean = float(metric["mean"])
	std = metric.get("std")
	return f"{mean:.4f}" if std is None else f"{mean:.4f} ± {float(std):.4f}"


def _csv_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
	return [
		{
			"Model": row["Model"],
			"Seeds": row["Seeds"],
			**{label: _format_metric(row["metrics"][label]) for label in METRIC_COLUMNS},
		}
		for row in rows
	]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=["Model", "Seeds", *METRIC_COLUMNS])
		writer.writeheader()
		writer.writerows(rows)


def _write_text(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
	columns = ["Model", "Seeds", *METRIC_COLUMNS]
	widths = {column: max([len(column), *(len(str(row[column])) for row in rows)]) for column in columns}
	lines = ["  ".join(column.ljust(widths[column]) for column in columns)]
	lines.append("  ".join("-" * widths[column] for column in columns))
	for row in rows:
		lines.append("  ".join(str(row[column]).ljust(widths[column]) for column in columns))
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tex_escape(value: str) -> str:
	return value.replace("_", "\\_")


def _write_tex(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
	lines = [
		"\\begin{table*}[t]",
		"\\centering",
		"\\small",
		"\\begin{tabular}{lcccccc}",
		"\\hline",
		"Model & Dice & IoU & Surface MAE & Canopy MAE & Energy Log MAE & Active Canopy MAE \\\\",
		"\\hline",
	]
	for index, row in enumerate(rows):
		if index == 5:
			lines.append("\\hline")
		values = []
		for label in METRIC_COLUMNS:
			metric = row["metrics"][label]
			if metric.get("std") is None:
				values.append(f"{float(metric['mean']):.4f}")
			else:
				values.append(f"${float(metric['mean']):.4f} \\pm {float(metric['std']):.4f}$")
		lines.append(_tex_escape(str(row["Model"])) + " & " + " & ".join(values) + " \\\\")
	lines.extend(["\\hline", "\\end{tabular}", "\\end{table*}"])
	path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(
	*,
	registry_path: Path = REGISTRY_PATH,
	output_root: Path = OUTPUT_ROOT,
	cawfe_root: Path = CAWFE_ROOT,
	summary_root: Path = SUMMARY_ROOT,
	evaluate_missing_cawfe: bool = True,
	allow_incomplete: bool = False,
) -> dict[str, Any]:
	registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
	external, missing_external = _discover_external_runs(registry, output_root)
	cawfe, missing_cawfe = _ensure_cawfe_test_runs(registry, cawfe_root, evaluate_missing=evaluate_missing_cawfe)
	all_runs = {**external, **cawfe}
	missing = [*missing_external, *missing_cawfe]
	if missing and not allow_incomplete:
		raise RuntimeError(
			"Table 2 aggregation refuses partial held-out evaluation. Missing runs:\n"
			+ "\n".join(f"- {item['model']} seed={item['seed']}: {item['reason']}" for item in missing)
		)
	display_names = {
		**{key: str(value["display_name"]) for key, value in registry["baselines"].items()},
		**{key: str(value["display_name"]) for key, value in registry["cawfe_latte_rows"].items()},
	}
	rows = [
		_aggregate_model(key, display_names[key], all_runs[key])
		for key in ROW_ORDER
		if all_runs.get(key)
	]
	payload = {
		"schema_version": 1,
		"row_order": list(ROW_ORDER),
		"test_based_model_selection": False,
		"frozen_final_model": "GA-Q2",
		"complete": not missing and len(rows) == len(ROW_ORDER),
		"missing_runs": missing,
		"rows": rows,
	}
	summary_root.mkdir(parents=True, exist_ok=True)
	_json_write(summary_root / "table2_results.json", payload)
	display_rows = _csv_rows(rows)
	_write_csv(summary_root / "table2_results.csv", display_rows)
	_write_text(summary_root / "table2_results.txt", display_rows)
	_write_tex(summary_root / "table2_results.tex", rows)
	return payload


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
	parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
	parser.add_argument("--cawfe-root", type=Path, default=CAWFE_ROOT)
	parser.add_argument("--summary-root", type=Path, default=SUMMARY_ROOT)
	parser.add_argument("--no-evaluate-missing-cawfe", action="store_true")
	parser.add_argument("--allow-incomplete", action="store_true")
	args = parser.parse_args()
	payload = summarize(
		registry_path=args.registry,
		output_root=args.output_root,
		cawfe_root=args.cawfe_root,
		summary_root=args.summary_root,
		evaluate_missing_cawfe=not args.no_evaluate_missing_cawfe,
		allow_incomplete=args.allow_incomplete,
	)
	print(json.dumps({"complete": payload["complete"], "rows": len(payload["rows"]), "missing": len(payload["missing_runs"])}, indent=2))


if __name__ == "__main__":
	main()
