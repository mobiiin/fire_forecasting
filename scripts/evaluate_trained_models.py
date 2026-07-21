"""Evaluate the best trained checkpoints and write paper-ready metrics."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import logging
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

from src.evaluation.run_discovery import TrainingRun, discover_runs, find_best_run, metric_value_for_run


DEFAULT_ARCHITECTURES = [
	"convlstm_unet",
	"earthformer_lite",
	"cawfe_st_mamba",
	"weatherformer_lite",
	"cawfe_latte_lite",
	"cawfe_latte",
]
OPTIONAL_ARCHITECTURES = [
	"current_frame_unet",
	"temporal_3d_unet",
	"convgru_unet",
	"attention_convlstm_unet",
	"swin_unet",
	"fno_2d",
	"ufno",
]
ARCHITECTURE_ALIASES = {
	"cawfe_st_mamba": "st_mamba_lite",
}
DISPLAY_NAMES = {
	"persistence": "Persistence",
	"linear_extrapolation": "Linear Extrapolation",
	"convlstm_unet": "ConvLSTM U-Net",
	"earthformer_lite": "Earthformer-lite",
	"st_mamba_lite": "CAWFE-ST-Mamba",
	"cawfe_st_mamba": "CAWFE-ST-Mamba",
	"weatherformer_lite": "WeatherFormer-lite",
	"cawfe_latte_lite": "CAWFE-Latte-Lite",
	"cawfe_latte": "CAWFE-Latte",
}

PAPER_COLUMNS = [
	"Model",
	"Surf. MAE ↓",
	"Canopy MAE ↓",
	"Dice ↑",
	"IoU ↑",
	"Energy MAE ↓",
	"Active Energy MAE ↓",
	"Skill ↑",
]
PER_FIRE_COLUMNS = [
	"architecture",
	"run_name",
	"fire_name",
	"split",
	"surf_mae",
	"canopy_mae",
	"dice",
	"iou",
	"energy_mae",
	"active_energy_mae",
	"skill",
	"num_samples",
	"checkpoint_path",
]
LATEX_HEADER = (
	"Model & Surf. MAE $\\downarrow$ & Canopy MAE $\\downarrow$ & Dice $\\uparrow$ & "
	"IoU $\\uparrow$ & Energy MAE $\\downarrow$ & Active Energy MAE $\\downarrow$ & Skill $\\uparrow$ \\\\"
)
LOWER_IS_BETTER_COLUMNS = {"Surf. MAE ↓", "Canopy MAE ↓", "Energy MAE ↓", "Active Energy MAE ↓"}
HIGHER_IS_BETTER_COLUMNS = {"Dice ↑", "IoU ↑", "Skill ↑"}


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Evaluate trained wildfire forecasting checkpoints for paper results.")
	parser.add_argument("--config", default="configs/default.yaml", help="Base YAML config used when a run lacks a resolved config.")
	parser.add_argument("--mode", choices=("quantitative", "qualitative"), default="quantitative", help="Evaluation mode.")
	parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="Dataset split to evaluate.")
	parser.add_argument("--model_architecture", default="all", help="Architecture to evaluate, or 'all'.")
	parser.add_argument("--runs_root", default="artifacts/runs", help="Root containing artifacts/runs/<architecture>/<run_name>.")
	parser.add_argument("--output_dir", default="artifacts/results", help="Root directory for evaluation outputs.")
	parser.add_argument("--selection_metric", default="best_metric", help="Run metric used to select the best run per architecture.")
	parser.add_argument("--selection_mode", choices=("auto", "min", "max"), default="auto", help="Run-selection direction.")
	parser.add_argument("--checkpoint_name", default="best_model.pt", help="Checkpoint filename under each run's checkpoints directory.")
	parser.add_argument("--max_batches", type=int, default=None, help="Optional debug cap on evaluated batches.")
	parser.add_argument("--num_workers", type=int, default=None, help="Override train/val/test DataLoader workers during evaluation.")
	parser.add_argument("--save_predictions", action="store_true", help="Save raw predictions when the selected evaluator supports it.")
	parser.add_argument(
		"--allow_sequence_mismatch",
		action="store_true",
		help="Allow evaluating a checkpoint whose saved T/horizon metadata does not match the evaluation config.",
	)
	parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing timestamped evaluation directory.")
	parser.add_argument(
		"--include_baselines",
		"--include-baselines",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Include persistence and linear-extrapolation baselines.",
	)
	parser.add_argument("--baseline_results_dir", default="artifacts/logs", help="Directory with existing baseline CSVs.")
	parser.add_argument(
		"--latex_bold_best",
		"--latex-bold-best",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Bold best values in LaTeX output.",
	)
	return parser


def canonical_architecture(name: str) -> str:
	return ARCHITECTURE_ALIASES.get(str(name).lower(), str(name).lower())


def display_name(name: str) -> str:
	key = canonical_architecture(name)
	return DISPLAY_NAMES.get(key, DISPLAY_NAMES.get(str(name).lower(), str(name).replace("_", " ").title()))


def selected_architectures(model_architecture: str) -> list[str]:
	if str(model_architecture).lower() == "all":
		return list(DEFAULT_ARCHITECTURES) + list(OPTIONAL_ARCHITECTURES)
	return [str(model_architecture).lower()]


def _json_default(value: Any) -> Any:
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, float) and not math.isfinite(value):
		return None
	return str(value)


def _to_jsonable(value: Any) -> Any:
	if isinstance(value, Mapping):
		return {str(key): _to_jsonable(nested) for key, nested in value.items()}
	if isinstance(value, list):
		return [_to_jsonable(item) for item in value]
	if isinstance(value, tuple):
		return [_to_jsonable(item) for item in value]
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, float) and not math.isfinite(value):
		return None
	return value


def _write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True, default=_json_default, allow_nan=False)


def _csv_cell(value: Any) -> Any:
	if isinstance(value, float) and not math.isfinite(value):
		return ""
	return value


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str] | None = None) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if fieldnames is None:
		fieldnames = []
		seen = set()
		for row in rows:
			for key in row.keys():
				if key not in seen:
					seen.add(key)
					fieldnames.append(str(key))
	with path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
		writer.writeheader()
		for row in rows:
			writer.writerow({key: _csv_cell(value) for key, value in dict(row).items()})


def _safe_float(value: Any) -> float:
	try:
		number = float(value)
	except (TypeError, ValueError):
		return math.nan
	return number if math.isfinite(number) else math.nan


def _metric(metrics: Mapping[str, Any], candidates: list[str]) -> tuple[float, str | None]:
	lookup = {str(key).lower(): value for key, value in metrics.items()}
	for candidate in candidates:
		value = _safe_float(lookup.get(candidate.lower()))
		if math.isfinite(value):
			return value, candidate
	return math.nan, None


def paper_values_from_metrics(metrics: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, str | None]]:
	values: dict[str, float] = {}
	sources: dict[str, str | None] = {}
	values["Surf. MAE ↓"], sources["Surf. MAE ↓"] = _metric(metrics, ["test_surface_consumed_mae", "surface_consumed_mae"])
	values["Canopy MAE ↓"], sources["Canopy MAE ↓"] = _metric(metrics, ["test_canopy_consumed_mae", "canopy_consumed_mae"])
	values["Dice ↑"], sources["Dice ↑"] = _metric(metrics, ["test_mask_dice", "mask_dice", "test_dice", "dice"])
	values["IoU ↑"], sources["IoU ↑"] = _metric(metrics, ["test_mask_iou", "mask_iou", "test_iou", "iou"])
	values["Energy MAE ↓"], sources["Energy MAE ↓"] = _metric(metrics, ["test_energy_mw_mae", "test_energy_MW_mae", "energy_MW_mae", "test_energy_log_mae", "energy_log_mae"])
	values["Active Energy MAE ↓"], sources["Active Energy MAE ↓"] = _metric(
		metrics,
		["test_energy_mw_active_mae", "test_energy_MW_active_mae", "energy_MW_active_mae", "test_active_mae", "active_mae"],
	)
	values["Skill ↑"] = math.nan
	sources["Skill ↑"] = None
	return values, sources


def _skill_error_basis(values: Mapping[str, float]) -> float:
	active_energy = _safe_float(values.get("Active Energy MAE ↓"))
	if math.isfinite(active_energy):
		return active_energy
	energy = _safe_float(values.get("Energy MAE ↓"))
	if math.isfinite(energy):
		return energy
	return math.nan


def apply_skill(values: dict[str, float], persistence_values: Mapping[str, float] | None, is_persistence: bool = False) -> None:
	if is_persistence:
		values["Skill ↑"] = 0.0
		return
	if persistence_values is None:
		values["Skill ↑"] = math.nan
		return
	model_error = _skill_error_basis(values)
	persistence_error = _skill_error_basis(persistence_values)
	if not math.isfinite(model_error) or not math.isfinite(persistence_error) or persistence_error == 0.0:
		values["Skill ↑"] = math.nan
		return
	values["Skill ↑"] = 1.0 - (model_error / persistence_error)


def build_paper_row(model_name: str, values: Mapping[str, float]) -> dict[str, Any]:
	row: dict[str, Any] = {"Model": model_name}
	for column in PAPER_COLUMNS[1:]:
		row[column] = values.get(column, math.nan)
	return row


def _wide_row(
	*,
	architecture: str,
	model_name: str,
	run_name: str,
	split: str,
	checkpoint_path: str,
	status: str,
	num_samples: int,
	values: Mapping[str, float],
	sources: Mapping[str, str | None],
) -> dict[str, Any]:
	row = build_paper_row(model_name, values)
	row.update(
		{
			"architecture": architecture,
			"run_name": run_name,
			"split": split,
			"checkpoint_path": checkpoint_path,
			"status": status,
			"num_samples": num_samples,
		}
	)
	for column, source in sources.items():
		row[f"{column} source"] = source
	return row


def _per_fire_rows(
	*,
	architecture: str,
	model_name: str,
	run_name: str,
	split: str,
	checkpoint_path: str,
	result: Mapping[str, Any],
	persistence_by_fire: Mapping[str, Mapping[str, float]] | None,
	is_persistence: bool,
) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	per_dataset = result.get("per_dataset_results", {})
	if not isinstance(per_dataset, Mapping):
		return rows
	for fire_name, metrics in sorted(per_dataset.items()):
		if not isinstance(metrics, Mapping):
			continue
		values, _sources = paper_values_from_metrics(metrics)
		persistence_values = None if persistence_by_fire is None else persistence_by_fire.get(str(fire_name))
		apply_skill(values, persistence_values, is_persistence=is_persistence)
		rows.append(
			{
				"architecture": architecture,
				"run_name": run_name,
				"fire_name": str(fire_name),
				"split": split,
				"surf_mae": values["Surf. MAE ↓"],
				"canopy_mae": values["Canopy MAE ↓"],
				"dice": values["Dice ↑"],
				"iou": values["IoU ↑"],
				"energy_mae": values["Energy MAE ↓"],
				"active_energy_mae": values["Active Energy MAE ↓"],
				"skill": values["Skill ↑"],
				"num_samples": int(_safe_float(metrics.get("num_samples")) if math.isfinite(_safe_float(metrics.get("num_samples"))) else 0),
				"checkpoint_path": checkpoint_path,
			}
		)
	return rows


def _long_metric_rows(
	*,
	architecture: str,
	model_name: str,
	run_name: str,
	split: str,
	checkpoint_path: str,
	result: Mapping[str, Any],
) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	aggregate = result.get("aggregate_results", {})
	if isinstance(aggregate, Mapping):
		for metric_name, value in sorted(aggregate.items()):
			rows.append(
				{
					"architecture": architecture,
					"model": model_name,
					"run_name": run_name,
					"split": split,
					"scope": "aggregate",
					"fire_name": "",
					"metric": metric_name,
					"value": value,
					"checkpoint_path": checkpoint_path,
				}
			)
	per_dataset = result.get("per_dataset_results", {})
	if isinstance(per_dataset, Mapping):
		for fire_name, metrics in sorted(per_dataset.items()):
			if not isinstance(metrics, Mapping):
				continue
			for metric_name, value in sorted(metrics.items()):
				rows.append(
					{
						"architecture": architecture,
						"model": model_name,
						"run_name": run_name,
						"split": split,
						"scope": "per_fire",
						"fire_name": fire_name,
						"metric": metric_name,
						"value": value,
						"checkpoint_path": checkpoint_path,
					}
				)
	return rows


def _best_columns(rows: list[Mapping[str, Any]]) -> dict[str, float]:
	best: dict[str, float] = {}
	for column in PAPER_COLUMNS[1:]:
		values = [_safe_float(row.get(column)) for row in rows]
		values = [value for value in values if math.isfinite(value)]
		if not values:
			continue
		best[column] = min(values) if column in LOWER_IS_BETTER_COLUMNS else max(values)
	return best


def _format_latex_value(column: str, value: Any, best_values: Mapping[str, float], bold_best: bool) -> str:
	number = _safe_float(value)
	if not math.isfinite(number):
		return "--"
	text = f"{number:.3f}" if column in HIGHER_IS_BETTER_COLUMNS else f"{number:.4g}"
	best = best_values.get(column)
	if bold_best and best is not None and math.isclose(number, best, rel_tol=1e-9, abs_tol=1e-12):
		return f"\\textbf{{{text}}}"
	return text


def render_latex_table(rows: list[Mapping[str, Any]], bold_best: bool = True) -> str:
	best_values = _best_columns(rows)
	lines = [
		"\\begin{tabular}{lrrrrrrr}",
		"\\toprule",
		LATEX_HEADER,
		"\\midrule",
	]
	for row in rows:
		cells = [str(row.get("Model", ""))]
		for column in PAPER_COLUMNS[1:]:
			cells.append(_format_latex_value(column, row.get(column), best_values, bold_best))
		lines.append(" & ".join(cells) + " \\\\")
	lines.extend(["\\bottomrule", "\\end{tabular}", ""])
	return "\n".join(lines)


def _make_output_dir(output_root: Path, mode: str, split: str, model_architecture: str, overwrite: bool) -> Path:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	prefix = f"{mode}_{split}"
	if str(model_architecture).lower() != "all":
		prefix = f"{prefix}_{canonical_architecture(model_architecture)}"
	base_dir = output_root / mode / f"{prefix}_{timestamp}"
	if overwrite or not base_dir.exists():
		base_dir.mkdir(parents=True, exist_ok=True)
		return base_dir
	for suffix in range(2, 1000):
		candidate = base_dir.with_name(f"{base_dir.name}_v{suffix}")
		if not candidate.exists():
			candidate.mkdir(parents=True, exist_ok=False)
			return candidate
	raise FileExistsError(f"Could not create a unique evaluation output directory under {output_root / mode}.")


def _setup_logger(output_dir: Path) -> logging.Logger:
	log_dir = output_dir / "logs"
	log_dir.mkdir(parents=True, exist_ok=True)
	logger = logging.getLogger("evaluate_trained_models")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
	file_handler = logging.FileHandler(log_dir / "evaluate_trained_models.log", encoding="utf-8")
	file_handler.setFormatter(formatter)
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)
	return logger


def _num_worker_override(num_workers: int | None) -> dict[str, Any]:
	if num_workers is None:
		return {}
	return {
		"num_workers": int(num_workers),
		"training": {"num_workers": int(num_workers)},
		"data_loader": {split: {"num_workers": int(num_workers)} for split in ("train", "val", "test")},
	}


def _model_config_override(architecture: str, checkpoint_path: str, num_workers: int | None, run_dir: str | Path | None = None) -> dict[str, Any]:
	override: dict[str, Any] = {
		"model": {"architecture": architecture, "name": architecture},
		"checkpoint": {"path": checkpoint_path, "best_path": checkpoint_path},
	}
	override.update(_num_worker_override(num_workers))
	if run_dir is not None:
		normalization_stats_path = Path(run_dir).expanduser().resolve() / "metadata" / "normalization_stats.npz"
		if normalization_stats_path.exists():
			override["normalization"] = {"path": str(normalization_stats_path)}
	return override


def _load_baseline_csv(path: Path, method_name: str, split: str) -> dict[str, Any] | None:
	if not path.exists():
		return None
	with path.open("r", newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle))
	aggregate_results: dict[str, float] | None = None
	per_dataset_results: dict[str, dict[str, float]] = {}
	for row in rows:
		if str(row.get("method", method_name)).lower() != method_name:
			continue
		if str(row.get("split", split)).lower() != split:
			continue
		numeric = {key: _safe_float(value) for key, value in row.items() if key not in {"method", "split", "scope", "dataset_name"}}
		numeric = {key: value for key, value in numeric.items() if math.isfinite(value)}
		scope = str(row.get("scope", "aggregate")).lower()
		if scope == "aggregate":
			aggregate_results = numeric
		elif scope == "per_fire":
			dataset_name = str(row.get("dataset_name", ""))
			if dataset_name:
				per_dataset_results[dataset_name] = numeric
	if aggregate_results is None:
		return None
	return {
		"method": method_name,
		"split": split,
		"num_samples": int(aggregate_results.get("num_samples", 0.0)),
		"aggregate_results": aggregate_results,
		"per_dataset_results": per_dataset_results,
		"rows": rows,
		"loaded_from": str(path),
	}


def _evaluate_or_load_baseline(
	*,
	method_name: str,
	config_path: str,
	split: str,
	baseline_results_dir: Path,
	output_dir: Path,
	max_batches: int | None,
	num_workers: int | None,
	save_predictions: bool,
	logger: logging.Logger,
) -> dict[str, Any]:
	existing_path = baseline_results_dir / f"{method_name}_baseline_{split}.csv"
	loaded = _load_baseline_csv(existing_path, method_name, split)
	if loaded is not None:
		logger.info("Loaded existing %s baseline results from %s", method_name, existing_path)
		return loaded
	from src.baselines import evaluate_baseline, predict_linear_extrapolation_for_sample, predict_persistence_for_sample

	predict_fn = predict_persistence_for_sample if method_name == "persistence" else predict_linear_extrapolation_for_sample
	output_csv = output_dir / "logs" / f"{method_name}_baseline_{split}.csv"
	logger.info("Running %s baseline on split=%s", method_name, split)
	return evaluate_baseline(
		config_path=config_path,
		split=split,
		method_name=method_name,
		predict_fn=predict_fn,
		mode="patch",
		max_batches=max_batches,
		config_override=_num_worker_override(num_workers),
		output_csv=output_csv,
		save_predictions=save_predictions,
		save_visualizations=False,
	)


def _copy_selected_run_metadata(selected_runs: list[TrainingRun], output_dir: Path) -> None:
	destination_root = output_dir / "selected_run_summaries"
	for run in selected_runs:
		if not run.summary_path:
			continue
		source = Path(run.summary_path)
		if not source.exists():
			continue
		destination = destination_root / run.architecture / run.run_name / source.name
		destination.parent.mkdir(parents=True, exist_ok=True)
		shutil.copy2(source, destination)


def _result_identity(architecture: str, run_name: str, checkpoint_path: str) -> dict[str, str]:
	return {
		"architecture": architecture,
		"model_name": display_name(architecture),
		"run_name": run_name,
		"checkpoint_path": checkpoint_path,
	}


def _result_rows(
	*,
	identity: Mapping[str, str],
	result: Mapping[str, Any],
	split: str,
	status: str,
	persistence_values: Mapping[str, float] | None,
	persistence_by_fire: Mapping[str, Mapping[str, float]] | None,
	is_persistence: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
	aggregate = result.get("aggregate_results", {})
	if not isinstance(aggregate, Mapping):
		aggregate = {}
	values, sources = paper_values_from_metrics(aggregate)
	apply_skill(values, persistence_values, is_persistence=is_persistence)
	paper_row = build_paper_row(identity["model_name"], values)
	wide_row = _wide_row(
		architecture=identity["architecture"],
		model_name=identity["model_name"],
		run_name=identity["run_name"],
		split=split,
		checkpoint_path=identity["checkpoint_path"],
		status=status,
		num_samples=int(result.get("num_samples", 0)),
		values=values,
		sources=sources,
	)
	per_fire = _per_fire_rows(
		architecture=identity["architecture"],
		model_name=identity["model_name"],
		run_name=identity["run_name"],
		split=split,
		checkpoint_path=identity["checkpoint_path"],
		result=result,
		persistence_by_fire=persistence_by_fire,
		is_persistence=is_persistence,
	)
	long_rows = _long_metric_rows(
		architecture=identity["architecture"],
		model_name=identity["model_name"],
		run_name=identity["run_name"],
		split=split,
		checkpoint_path=identity["checkpoint_path"],
		result=result,
	)
	metadata = {
		"identity": dict(identity),
		"sequence": dict(result.get("sequence", {})) if isinstance(result.get("sequence", {}), Mapping) else {},
		"paper_values": values,
		"metric_sources": sources,
		"num_samples": int(result.get("num_samples", 0)),
	}
	return paper_row, wide_row, per_fire, long_rows, metadata


def run_qualitative_mode() -> None:
	raise NotImplementedError(
		"Qualitative prediction and rollout visualization will be implemented later. "
		"Use --mode quantitative for paper metrics."
	)


def run_quantitative(args: argparse.Namespace) -> Path:
	output_root = Path(args.output_dir).expanduser().resolve()
	output_dir = _make_output_dir(output_root, "quantitative", args.split, args.model_architecture, bool(args.overwrite))
	logger = _setup_logger(output_dir)
	if args.save_predictions:
		logger.warning("Model prediction saving is not implemented in the shared quantitative checkpoint evaluator yet.")

	requested = selected_architectures(args.model_architecture)
	all_model_mode = str(args.model_architecture).lower() == "all"
	runs_root = Path(args.runs_root).expanduser().resolve()
	all_runs = discover_runs(runs_root, checkpoint_name=args.checkpoint_name)
	selected_runs: list[TrainingRun] = []
	selected_rows: list[dict[str, Any]] = []
	skipped_models: list[str] = []
	failed_models: list[dict[str, str]] = []
	successful_models: list[str] = []
	paper_rows: list[dict[str, Any]] = []
	wide_rows: list[dict[str, Any]] = []
	per_fire_rows: list[dict[str, Any]] = []
	long_rows: list[dict[str, Any]] = []
	json_entries: list[dict[str, Any]] = []
	persistence_values: dict[str, float] | None = None
	persistence_by_fire: dict[str, dict[str, float]] | None = None

	logger.info("Evaluation output directory: %s", output_dir)
	logger.info("Discovering runs under %s", runs_root)

	if bool(args.include_baselines):
		for method_name in ("persistence", "linear_extrapolation"):
			try:
				result = _evaluate_or_load_baseline(
					method_name=method_name,
					config_path=args.config,
					split=args.split,
					baseline_results_dir=Path(args.baseline_results_dir).expanduser().resolve(),
					output_dir=output_dir,
					max_batches=args.max_batches,
					num_workers=args.num_workers,
					save_predictions=bool(args.save_predictions),
					logger=logger,
				)
				identity = _result_identity(method_name, method_name, "")
				paper_row, wide_row, baseline_per_fire, baseline_long, metadata = _result_rows(
					identity=identity,
					result=result,
					split=args.split,
					status="completed",
					persistence_values=persistence_values,
					persistence_by_fire=persistence_by_fire,
					is_persistence=method_name == "persistence",
				)
				paper_rows.append(paper_row)
				wide_rows.append(wide_row)
				per_fire_rows.extend(baseline_per_fire)
				long_rows.extend(baseline_long)
				json_entries.append(metadata)
				if method_name == "persistence":
					persistence_values, _sources = paper_values_from_metrics(result.get("aggregate_results", {}))
					apply_skill(persistence_values, None, is_persistence=True)
					persistence_by_fire = {}
					for row in baseline_per_fire:
						persistence_by_fire[str(row["fire_name"])] = {
							"Active Energy MAE ↓": _safe_float(row.get("active_energy_mae")),
							"Energy MAE ↓": _safe_float(row.get("energy_mae")),
						}
			except Exception as exc:
				logger.exception("Baseline %s failed", method_name)
				failed_models.append({"architecture": method_name, "error": str(exc)})
				if not all_model_mode:
					raise

	for requested_architecture in requested:
		architecture = canonical_architecture(requested_architecture)
		runs = [run for run in all_runs if run.architecture.lower() == architecture]
		best_run = find_best_run(runs, architecture, selection_metric=args.selection_metric, selection_mode=args.selection_mode)
		if best_run is None:
			message = f"No trained runs found for {requested_architecture}; skipping."
			if all_model_mode:
				logger.info(message)
				print(message)
				skipped_models.append(requested_architecture)
				continue
			raise FileNotFoundError(f"No trained runs found for requested architecture {requested_architecture!r} under {runs_root}.")
		selected_runs.append(best_run)
		selected_value, selected_metric = metric_value_for_run(best_run, args.selection_metric)
		selected_rows.append(
			{
				"architecture": architecture,
				"requested_architecture": requested_architecture,
				"selected_run": best_run.run_name,
				"status": best_run.status,
				"best_epoch": best_run.best_epoch,
				"selected_metric": selected_metric,
				"selected_metric_value": selected_value,
				"checkpoint_path": best_run.checkpoint_path,
				"resolved_config_path": best_run.resolved_config_path,
			}
		)
		logger.info(
			"Selected %s | run=%s | metric=%s | value=%s | checkpoint=%s",
			architecture,
			best_run.run_name,
			selected_metric,
			selected_value,
			best_run.checkpoint_path,
		)
		try:
			config_path = best_run.resolved_config_path or args.config
			from src.models.evaluation import evaluate_checkpoint_on_split

			result = evaluate_checkpoint_on_split(
				config_path=config_path,
				split=args.split,
				checkpoint_path=best_run.checkpoint_path,
				checkpoint_kind="best",
				config_override=_model_config_override(architecture, best_run.checkpoint_path, args.num_workers, run_dir=best_run.run_dir),
				max_batches=args.max_batches,
				expected_architecture=architecture,
				allow_sequence_mismatch=bool(args.allow_sequence_mismatch),
			)
			identity = _result_identity(architecture, best_run.run_name, best_run.checkpoint_path)
			paper_row, wide_row, model_per_fire, model_long, metadata = _result_rows(
				identity=identity,
				result=result,
				split=args.split,
				status="completed",
				persistence_values=persistence_values,
				persistence_by_fire=persistence_by_fire,
			)
			paper_rows.append(paper_row)
			wide_rows.append(wide_row)
			per_fire_rows.extend(model_per_fire)
			long_rows.extend(model_long)
			json_entries.append(metadata)
			successful_models.append(architecture)
		except Exception as exc:
			logger.exception("Evaluation failed for %s", architecture)
			failed_models.append({"architecture": architecture, "run_name": best_run.run_name, "error": str(exc)})
			if not all_model_mode:
				raise

	_copy_selected_run_metadata(selected_runs, output_dir)

	_write_csv(output_dir / "selected_runs.csv", selected_rows)
	_write_json(output_dir / "selected_runs.json", selected_rows)
	_write_csv(output_dir / "paper_metrics.csv", paper_rows, PAPER_COLUMNS)
	_write_csv(output_dir / "paper_metrics_wide.csv", wide_rows)
	_write_json(output_dir / "paper_metrics.json", json_entries)
	_write_csv(output_dir / "per_fire_metrics.csv", per_fire_rows, PER_FIRE_COLUMNS)
	_write_csv(output_dir / "all_metrics_long.csv", long_rows)
	table_text = render_latex_table(paper_rows, bold_best=bool(args.latex_bold_best))
	(output_dir / "paper_table.tex").write_text(table_text, encoding="utf-8")
	summary = {
		"mode": "quantitative",
		"split": args.split,
		"config": str(Path(args.config).expanduser().resolve()),
		"runs_root": str(runs_root),
		"output_dir": str(output_dir),
		"selection_metric": args.selection_metric,
		"selection_mode": args.selection_mode,
		"allow_sequence_mismatch": bool(args.allow_sequence_mismatch),
		"successful_models": successful_models,
		"skipped_models": skipped_models,
		"failed_models": failed_models,
		"output_files": {
			"selected_runs_csv": str(output_dir / "selected_runs.csv"),
			"paper_metrics_csv": str(output_dir / "paper_metrics.csv"),
			"paper_metrics_wide_csv": str(output_dir / "paper_metrics_wide.csv"),
			"paper_table_tex": str(output_dir / "paper_table.tex"),
			"per_fire_metrics_csv": str(output_dir / "per_fire_metrics.csv"),
			"all_metrics_long_csv": str(output_dir / "all_metrics_long.csv"),
		},
	}
	_write_json(output_dir / "evaluation_summary.json", summary)

	if selected_rows:
		print("architecture | selected_run | best_epoch | selected_metric | checkpoint_path")
		print("-" * 100)
		for row in selected_rows:
			print(
				f"{row['architecture']} | {row['selected_run']} | {row['best_epoch']} | "
				f"{row['selected_metric']}={row['selected_metric_value']} | {row['checkpoint_path']}"
			)
	print(f"successful models: {', '.join(successful_models) if successful_models else 'none'}")
	print(f"skipped models: {', '.join(skipped_models) if skipped_models else 'none'}")
	print(f"failed models: {', '.join(item['architecture'] for item in failed_models) if failed_models else 'none'}")
	print(f"output_dir: {output_dir}")
	return output_dir


def main(argv: list[str] | None = None) -> None:
	args = build_argument_parser().parse_args(argv)
	if args.mode == "qualitative":
		run_qualitative_mode()
	run_quantitative(args)


if __name__ == "__main__":
	main()
