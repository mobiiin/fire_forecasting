"""Evaluate the best trained checkpoints and write paper-ready metrics."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
import math
import sys
from pathlib import Path
import shutil
from typing import Any, Mapping
import warnings

from src.config import load_config
from src.data.cache import (
	compute_dataset_index_hash,
	resolve_dataset_index_path,
	target_definition_version,
	temporal_target_offsets,
	validate_patch_cache,
)
from src.evaluation.run_discovery import TrainingRun, discover_runs, find_best_run, metric_value_for_run
from src.evaluation.qualitative import (
	coerce_chw4,
	load_qualitative_samples,
	save_individual_model_images,
	save_qualitative_summary_image,
	select_qualitative_sample_indices,
)
from src.training.checkpoints import load_checkpoint, load_model_state_dict_compatible
from src.training.input_normalization import normalization_config, resolve_input_normalization_stats_path
from src.training.model_outputs import extract_prediction
from src.training.batch_utils import unpack_batch
from src.training.train import _ensure_config_path


DEFAULT_ARCHITECTURES = [
	"convlstm_unet",
	"earthformer_lite",
	"st_mamba_lite",
	"weatherformer_lite",
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
BASELINE_ARCHITECTURES = ["persistence", "linear_extrapolation"]
REMOVED_MODEL_ARCHITECTURES = {"cawfe_latte_lite"}
REMOVED_MODEL_ARCHITECTURE_MESSAGE = "The old CAWFE-Latte-Lite implementation has been removed. A new design will be added later."
SUPPORTED_MODEL_ARCHITECTURES = {
	"all",
	"baseline",
	"baselines",
	"persistence",
	"linear_extrapolation",
	*DEFAULT_ARCHITECTURES,
	*ARCHITECTURE_ALIASES.keys(),
	"cawfe_latte",
}
DISPLAY_NAMES = {
	"persistence": "Persistence",
	"linear_extrapolation": "Linear Extrapolation",
	"convlstm_unet": "ConvLSTM U-Net",
	"earthformer_lite": "Earthformer-lite",
	"st_mamba_lite": "CAWFE-ST-Mamba",
	"cawfe_st_mamba": "CAWFE-ST-Mamba",
	"weatherformer_lite": "WeatherFormer-lite",
	"cawfe_latte": "CAWFE-Latte Encoders + Fusion",
}

PAPER_COLUMNS = [
	"Model",
	"Surf. MAE ↓",
	"Canopy MAE ↓",
	"Dice ↑",
	"IoU ↑",
	"Energy Log MAE ↓",
	"Active Energy Log MAE ↓",
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
	"energy_log_mae",
	"active_energy_log_mae",
	"energy_mw_mae",
	"active_energy_mw_mae",
	"paper_energy_mae",
	"paper_active_energy_mae",
	"skill",
	"num_samples",
	"checkpoint_path",
]
LATEX_HEADER = (
	"Model & Surf. MAE $\\downarrow$ & Canopy MAE $\\downarrow$ & Dice $\\uparrow$ & "
	"IoU $\\uparrow$ & Energy Log MAE $\\downarrow$ & Active Energy Log MAE $\\downarrow$ & Skill $\\uparrow$ \\\\"
)
LOWER_IS_BETTER_COLUMNS = {"Surf. MAE ↓", "Canopy MAE ↓", "Energy Log MAE ↓", "Active Energy Log MAE ↓", "Energy MW MAE ↓", "Active Energy MW MAE ↓"}
HIGHER_IS_BETTER_COLUMNS = {"Dice ↑", "IoU ↑", "Skill ↑"}


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Evaluate trained wildfire forecasting checkpoints for paper results.")
	parser.add_argument("--config", default="configs/default.yaml", help="Base YAML config used when a run lacks a resolved config.")
	parser.add_argument("--mode", choices=("quantitative", "qualitative", "all"), default="quantitative", help="Evaluation mode. Use 'all' to run quantitative then qualitative.")
	parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="Dataset split to evaluate.")
	parser.add_argument("--model_architecture", default="all", help="Architecture to evaluate, or 'all'.")
	parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path. Only valid with one learned architecture.")
	parser.add_argument("--runs_root", default="artifacts/runs", help="Root containing artifacts/runs/<architecture>/<run_name>.")
	parser.add_argument("--output_dir", default="artifacts/results", help="Root directory for evaluation outputs.")
	parser.add_argument("--eval_name", default="auto", help="Evaluation output folder name, or 'auto'.")
	parser.add_argument("--selection_metric", default="best_metric", help="Run metric used to select the best run per architecture.")
	parser.add_argument("--selection_mode", choices=("auto", "min", "max"), default="auto", help="Run-selection direction.")
	parser.add_argument("--checkpoint_name", default="best_model.pt", help="Checkpoint filename under each run's checkpoints directory.")
	parser.add_argument("--max_batches", type=int, default=None, help="Optional debug cap on evaluated batches.")
	parser.add_argument("--num_workers", type=int, default=None, help="Override train/val/test DataLoader workers during evaluation.")
	parser.add_argument("--num_samples", type=int, default=10, help="Number of randomly selected samples for qualitative mode.")
	parser.add_argument("--qualitative_seed", type=int, default=42, help="Random seed for qualitative sample selection.")
	parser.add_argument(
		"--qualitative_output_format",
		choices=("png", "pdf"),
		default="png",
		help="Image format for qualitative summary figures.",
	)
	parser.add_argument("--qualitative_dpi", type=int, default=180, help="DPI for qualitative summary figures.")
	parser.add_argument(
		"--save_individual_model_panels",
		action="store_true",
		default=False,
		help="Also save one single-model panel per model/sample in qualitative mode.",
	)
	parser.add_argument(
		"--paper_energy_metric",
		choices=("log", "mw"),
		default="log",
		help="Energy metric basis for paper table and skill. Default: log.",
	)
	parser.add_argument("--save_predictions", action="store_true", help="Save raw predictions when the selected evaluator supports it.")
	parser.add_argument(
		"--allow_sequence_mismatch",
		action="store_true",
		help="Allow evaluating a checkpoint whose saved T/horizon metadata does not match the evaluation config.",
	)
	parser.add_argument(
		"--allow_normalization_mismatch",
		action="store_true",
		help="Allow evaluating a checkpoint whose saved input-normalization metadata differs from the evaluation config.",
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
	parser.set_defaults(include_csv_outputs=False, write_latex=False, include_json_outputs=False)
	parser.add_argument("--include_csv_outputs", action="store_true", dest="include_csv_outputs", help="Write secondary CSV outputs.")
	parser.add_argument("--no_csv_outputs", action="store_false", dest="include_csv_outputs", help="Skip secondary CSV outputs.")
	parser.add_argument("--include_json_outputs", action="store_true", dest="include_json_outputs", help="Write machine-readable JSON sidecars.")
	parser.add_argument("--no_json_outputs", action="store_false", dest="include_json_outputs", help="Skip machine-readable JSON sidecars.")
	parser.add_argument("--write_latex", action="store_true", dest="write_latex", help="Write paper_table.tex.")
	parser.add_argument("--no_latex", action="store_false", dest="write_latex", help="Skip paper_table.tex.")
	parser.add_argument("--verbose", action="store_true", help="Print extra progress details.")
	parser.add_argument(
		"--write_log_file",
		"--write-log-file",
		action=argparse.BooleanOptionalAction,
		default=False,
		help="Write logs/evaluate_trained_models.log in the evaluation output directory.",
	)
	parser.add_argument(
		"--copy_selected_run_metadata",
		"--copy-selected-run-metadata",
		action=argparse.BooleanOptionalAction,
		default=False,
		help="Copy selected training run summaries into the evaluation output directory.",
	)
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
	requested = str(model_architecture).lower()
	if requested in REMOVED_MODEL_ARCHITECTURES:
		raise ValueError(REMOVED_MODEL_ARCHITECTURE_MESSAGE)
	if requested == "all":
		return list(DEFAULT_ARCHITECTURES) + list(OPTIONAL_ARCHITECTURES)
	if requested in {"baseline", "baselines", "persistence", "linear_extrapolation"}:
		return []
	return [canonical_architecture(requested)]


def _resolve_requested_targets(model_architecture: str, include_baselines: bool = True) -> tuple[list[str], list[str], bool]:
	"""Resolve learned and baseline targets from the user-facing architecture selector."""

	requested = str(model_architecture).strip().lower()
	if requested in REMOVED_MODEL_ARCHITECTURES:
		raise ValueError(REMOVED_MODEL_ARCHITECTURE_MESSAGE)
	if requested not in SUPPORTED_MODEL_ARCHITECTURES and canonical_architecture(requested) not in DEFAULT_ARCHITECTURES + OPTIONAL_ARCHITECTURES:
		supported = ", ".join(sorted(SUPPORTED_MODEL_ARCHITECTURES))
		raise ValueError(f"Unsupported --model_architecture {model_architecture!r}. Supported values: {supported}")
	if requested == "all":
		return list(DEFAULT_ARCHITECTURES) + list(OPTIONAL_ARCHITECTURES), list(BASELINE_ARCHITECTURES) if include_baselines else [], True
	if requested in {"baseline", "baselines"}:
		return [], list(BASELINE_ARCHITECTURES), False
	if requested in BASELINE_ARCHITECTURES:
		return [], [requested], False
	return [canonical_architecture(requested)], [], False


def _validate_checkpoint_arg(checkpoint: str | None, learned_targets: list[str], baseline_targets: list[str], requested: str) -> None:
	if checkpoint in (None, "", "null"):
		return
	if baseline_targets or not learned_targets:
		raise ValueError("Baselines do not use learned checkpoints.")
	if len(learned_targets) != 1:
		raise ValueError("--checkpoint can only be used with a single learned model architecture.")
	if str(requested).lower() == "all":
		raise ValueError("--checkpoint can only be used with a single learned model architecture.")


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


def _energy_paper_columns(paper_energy_metric: str) -> tuple[str, str]:
	if str(paper_energy_metric).lower() == "mw":
		return "Energy MW MAE ↓", "Active Energy MW MAE ↓"
	return "Energy Log MAE ↓", "Active Energy Log MAE ↓"


def paper_columns_for_metric(paper_energy_metric: str) -> list[str]:
	energy_column, active_energy_column = _energy_paper_columns(paper_energy_metric)
	return ["Model", "Surf. MAE ↓", "Canopy MAE ↓", "Dice ↑", "IoU ↑", energy_column, active_energy_column, "Skill ↑"]


def paper_values_from_metrics(metrics: Mapping[str, Any], paper_energy_metric: str = "log") -> tuple[dict[str, float], dict[str, str | None]]:
	values: dict[str, float] = {}
	sources: dict[str, str | None] = {}
	values["Surf. MAE ↓"], sources["Surf. MAE ↓"] = _metric(metrics, ["test_surface_consumed_mae", "surface_consumed_mae"])
	values["Canopy MAE ↓"], sources["Canopy MAE ↓"] = _metric(metrics, ["test_canopy_consumed_mae", "canopy_consumed_mae"])
	values["Dice ↑"], sources["Dice ↑"] = _metric(metrics, ["test_mask_dice", "mask_dice", "test_dice", "dice"])
	values["IoU ↑"], sources["IoU ↑"] = _metric(metrics, ["test_mask_iou", "mask_iou", "test_iou", "iou"])
	energy_column, active_energy_column = _energy_paper_columns(paper_energy_metric)
	if str(paper_energy_metric).lower() == "mw":
		values[energy_column], sources[energy_column] = _metric(metrics, ["test_energy_mw_mae", "test_energy_MW_mae", "energy_mw_mae", "energy_MW_mae"])
		values[active_energy_column], sources[active_energy_column] = _metric(
			metrics,
			["test_active_energy_mw_mae", "test_energy_mw_active_mae", "test_energy_MW_active_mae", "active_energy_mw_mae", "energy_MW_active_mae"],
		)
	else:
		values[energy_column], sources[energy_column] = _metric(metrics, ["test_energy_log_mae", "energy_log_mae"])
		values[active_energy_column], sources[active_energy_column] = _metric(
			metrics,
			["test_active_energy_log_mae", "active_energy_log_mae"],
		)
	values["Skill ↑"] = math.nan
	sources["Skill ↑"] = None
	return values, sources


def _skill_error_basis(values: Mapping[str, float], paper_energy_metric: str = "log") -> float:
	energy_column, active_energy_column = _energy_paper_columns(paper_energy_metric)
	active_energy = _safe_float(values.get(active_energy_column))
	if math.isfinite(active_energy):
		return active_energy
	energy = _safe_float(values.get(energy_column))
	if math.isfinite(energy):
		return energy
	return math.nan


def apply_skill(
	values: dict[str, float],
	persistence_values: Mapping[str, float] | None,
	is_persistence: bool = False,
	paper_energy_metric: str = "log",
) -> None:
	if is_persistence:
		values["Skill ↑"] = 0.0
		return
	if persistence_values is None:
		values["Skill ↑"] = math.nan
		return
	model_error = _skill_error_basis(values, paper_energy_metric=paper_energy_metric)
	persistence_error = _skill_error_basis(persistence_values, paper_energy_metric=paper_energy_metric)
	if not math.isfinite(model_error) or not math.isfinite(persistence_error) or persistence_error == 0.0:
		values["Skill ↑"] = math.nan
		return
	values["Skill ↑"] = 1.0 - (model_error / persistence_error)


def build_paper_row(model_name: str, values: Mapping[str, float], paper_energy_metric: str = "log") -> dict[str, Any]:
	row: dict[str, Any] = {"Model": model_name}
	for column in paper_columns_for_metric(paper_energy_metric)[1:]:
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
	paper_energy_metric: str = "log",
) -> dict[str, Any]:
	row = build_paper_row(model_name, values, paper_energy_metric=paper_energy_metric)
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
	paper_energy_metric: str,
) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	per_dataset = result.get("per_dataset_results", {})
	if not isinstance(per_dataset, Mapping):
		return rows
	for fire_name, metrics in sorted(per_dataset.items()):
		if not isinstance(metrics, Mapping):
			continue
		values, _sources = paper_values_from_metrics(metrics, paper_energy_metric=paper_energy_metric)
		persistence_values = None if persistence_by_fire is None else persistence_by_fire.get(str(fire_name))
		apply_skill(values, persistence_values, is_persistence=is_persistence, paper_energy_metric=paper_energy_metric)
		energy_column, active_energy_column = _energy_paper_columns(paper_energy_metric)
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
				"energy_log_mae": values.get("Energy Log MAE ↓"),
				"active_energy_log_mae": values.get("Active Energy Log MAE ↓"),
				"energy_mw_mae": values.get("Energy MW MAE ↓"),
				"active_energy_mw_mae": values.get("Active Energy MW MAE ↓"),
				"paper_energy_mae": values.get(energy_column),
				"paper_active_energy_mae": values.get(active_energy_column),
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


def _best_columns(rows: list[Mapping[str, Any]], paper_energy_metric: str = "log") -> dict[str, float]:
	best: dict[str, float] = {}
	for column in paper_columns_for_metric(paper_energy_metric)[1:]:
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


def render_latex_table(rows: list[Mapping[str, Any]], bold_best: bool = True, paper_energy_metric: str = "log") -> str:
	best_values = _best_columns(rows, paper_energy_metric=paper_energy_metric)
	energy_column, active_energy_column = _energy_paper_columns(paper_energy_metric)
	latex_header = (
		"Model & Surf. MAE $\\downarrow$ & Canopy MAE $\\downarrow$ & Dice $\\uparrow$ & "
		f"IoU $\\uparrow$ & {energy_column.removesuffix(' ↓')} $\\downarrow$ & "
		f"{active_energy_column.removesuffix(' ↓')} $\\downarrow$ & Skill $\\uparrow$ \\\\"
	)
	lines = [
		"\\begin{tabular}{lrrrrrrr}",
		"\\toprule",
		latex_header,
		"\\midrule",
	]
	for row in rows:
		cells = [str(row.get("Model", ""))]
		for column in paper_columns_for_metric(paper_energy_metric)[1:]:
			cells.append(_format_latex_value(column, row.get(column), best_values, bold_best))
		lines.append(" & ".join(cells) + " \\\\")
	lines.extend(["\\bottomrule", "\\end{tabular}", ""])
	return "\n".join(lines)


def _make_output_dir(output_root: Path, mode: str, split: str, model_architecture: str, overwrite: bool, eval_name: str = "auto") -> Path:
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	if eval_name not in (None, "", "auto", "null"):
		base_dir = output_root / str(eval_name)
	else:
		prefix = f"{mode}_{split}_{str(model_architecture).lower()}"
		base_dir = output_root / f"{prefix}_{timestamp}"
	if overwrite or not base_dir.exists():
		base_dir.mkdir(parents=True, exist_ok=True)
		return base_dir
	for suffix in range(2, 1000):
		candidate = base_dir.with_name(f"{base_dir.name}_v{suffix}")
		if not candidate.exists():
			candidate.mkdir(parents=True, exist_ok=False)
			return candidate
	raise FileExistsError(f"Could not create a unique evaluation output directory under {output_root}.")


def _setup_logger(output_dir: Path, write_log_file: bool = False) -> logging.Logger:
	logger = logging.getLogger("evaluate_trained_models")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	if write_log_file:
		log_dir = output_dir / "logs"
		log_dir.mkdir(parents=True, exist_ok=True)
		file_handler = logging.FileHandler(log_dir / "evaluate_trained_models.log", encoding="utf-8")
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
	return logger


def _load_config_with_path(config_path: str | Path) -> dict[str, Any]:
	return _ensure_config_path(load_config(config_path), config_path)


def _sequence_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
	offsets = temporal_target_offsets(config)
	cache_config = config.get("cache", {}) if isinstance(config.get("cache"), Mapping) else {}
	index_path = resolve_dataset_index_path(config)
	return {
		"input_sequence_length": int(config["input_sequence_length"]),
		"prediction_horizon": int(config["prediction_horizon"]),
		"target_offset_from_start": int(offsets["target_offset_from_start"]),
		"target_offset_from_last_input": int(offsets["target_offset_from_last_input"]),
		"target_definition_version": target_definition_version(config),
		"cache_version": cache_config.get("cache_version"),
		"dataset_index": str(index_path) if index_path is not None else None,
		"dataset_index_hash": compute_dataset_index_hash(config),
	}


def _cache_metadata(config: Mapping[str, Any], split: str) -> dict[str, Any]:
	cache_config = config.get("cache", {}) if isinstance(config.get("cache"), Mapping) else {}
	if not bool(cache_config.get("enabled", False) and cache_config.get("use_precomputed_patches", False)):
		return {"enabled": False}
	cache_summary = validate_patch_cache(config, split=[split])
	manifest = cache_summary.get("manifest", {}) if isinstance(cache_summary.get("manifest"), Mapping) else {}
	return {
		"enabled": True,
		"cache_dir": cache_summary.get("cache_dir"),
		"manifest_path": cache_summary.get("manifest_path"),
		"cache_version": manifest.get("cache_version"),
		"input_sequence_length": manifest.get("input_sequence_length"),
		"prediction_horizon": manifest.get("prediction_horizon"),
		"target_offset_from_start": manifest.get("target_offset_from_start"),
		"target_definition_version": manifest.get("target_definition_version"),
		"dataset_index_hash": manifest.get("dataset_index_hash"),
		"split": dict(cache_summary.get("splits", {}).get(split, {})) if isinstance(cache_summary.get("splits"), Mapping) else {},
	}


def _normalization_setup_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
	normalization = normalization_config(config)
	stats_path = resolve_input_normalization_stats_path(config, must_exist=False)
	stats_channel_count = None
	fit_split = normalization.get("fit_split", "train")
	if stats_path is not None:
		try:
			from src.data.preprocessing import load_normalization_stats

			stats = load_normalization_stats(stats_path)
			if "mean" in stats:
				import numpy as np

				stats_channel_count = int(np.asarray(stats["mean"]).reshape(-1).shape[0])
			fit_split_value = stats.get("fit_split", stats.get("split_used")) if isinstance(stats, Mapping) else None
			if fit_split_value is not None:
				import numpy as np

				fit_split = str(np.asarray(fit_split_value).reshape(-1)[0])
		except Exception as exc:
			fit_split = f"unreadable: {exc}"
	return {
		"enabled": bool(normalization.get("enabled", bool(normalization))),
		"device": normalization.get("input_normalization_device", config.get("training", {}).get("input_normalization_device") if isinstance(config.get("training"), Mapping) else None),
		"stats_path": str(stats_path) if stats_path is not None else None,
		"fit_split": fit_split,
		"stats_channel_count": stats_channel_count,
		"apply_to_splits": normalization.get("apply_to_splits", ["train", "val", "test"]),
	}


def _checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
	path = Path(checkpoint_path).expanduser().resolve()
	try:
		checkpoint = load_checkpoint(path, map_location="cpu")
	except Exception as exc:
		return {"path": str(path), "read_error": str(exc)}
	keys = (
		"architecture",
		"run_name",
		"epoch",
		"best_epoch",
		"best_metric",
		"best_metric_name",
		"input_sequence_length",
		"prediction_horizon",
		"target_offset_from_start",
		"target_offset_from_last_input",
		"target_definition_version",
		"normalization",
	)
	return {"path": str(path), **{key: checkpoint.get(key) for key in keys if key in checkpoint}}


def _infer_run_dir_from_checkpoint(checkpoint_path: str | Path) -> Path | None:
	path = Path(checkpoint_path).expanduser().resolve()
	if path.parent.name == "checkpoints" and path.parent.parent.exists():
		return path.parent.parent
	return None


def _explicit_training_run(architecture: str, checkpoint_path: str | Path) -> TrainingRun:
	resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
	if not resolved_checkpoint.exists():
		raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")
	run_dir = _infer_run_dir_from_checkpoint(resolved_checkpoint)
	summary: dict[str, Any] = {}
	summary_path: Path | None = None
	if run_dir is not None:
		summary_path = run_dir / "metadata" / "run_summary.json"
		if summary_path.exists():
			try:
				summary = json.loads(summary_path.read_text(encoding="utf-8"))
			except Exception:
				summary = {}
	resolved_config_path = None
	config_path = None
	if run_dir is not None:
		resolved_candidate = run_dir / "configs" / "resolved_config.yaml"
		original_candidate = run_dir / "configs" / "original_config.yaml"
		resolved_config_path = str(resolved_candidate.resolve()) if resolved_candidate.exists() else None
		config_path = str(original_candidate.resolve()) if original_candidate.exists() else None
	return TrainingRun(
		architecture=architecture,
		run_name=str(summary.get("run_name") or (run_dir.name if run_dir is not None else resolved_checkpoint.stem)),
		run_dir=str(run_dir.resolve()) if run_dir is not None else str(resolved_checkpoint.parent),
		status=str(summary.get("status") or "explicit_checkpoint").lower(),
		checkpoint_path=str(resolved_checkpoint),
		config_path=config_path,
		resolved_config_path=resolved_config_path,
		best_metric_name=str(summary.get("best_metric_name")) if summary.get("best_metric_name") not in (None, "") else None,
		best_metric_value=_safe_float(summary.get("best_metric_value")),
		best_epoch=None if summary.get("best_epoch") in (None, "") else int(summary.get("best_epoch")),
		final_val_loss=_safe_float(summary.get("final_val_loss")),
		metadata_partial=not bool(summary),
		summary_path=str(summary_path.resolve()) if summary_path is not None and summary_path.exists() else None,
		metrics={},
	)


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
		metadata_dir = Path(run_dir).expanduser().resolve() / "metadata"
		for candidate in (
			metadata_dir / "normalization_stats.npz",
			metadata_dir / "normalization_stats.json",
		):
			if candidate.exists():
				override["normalization"] = {"path": str(candidate), "stats_path": str(candidate)}
				break
	return override


def _baseline_csv_sequence_matches(rows: list[Mapping[str, Any]], expected_sequence: Mapping[str, Any]) -> bool:
	aggregate_rows = [row for row in rows if str(row.get("scope", "aggregate")).lower() == "aggregate"]
	if not aggregate_rows:
		return False
	row = aggregate_rows[0]
	for key in ("input_sequence_length", "prediction_horizon", "target_offset_from_start", "target_definition_version"):
		if key not in row or row.get(key) in (None, ""):
			return False
		if str(row.get(key)) != str(expected_sequence.get(key)):
			return False
	return True


def _load_baseline_csv(path: Path, method_name: str, split: str, expected_sequence: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
	if not path.exists():
		return None
	with path.open("r", newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle))
	if expected_sequence is not None and not _baseline_csv_sequence_matches(rows, expected_sequence):
		return None
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
		"sequence": dict(expected_sequence or {}),
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
	expected_sequence: Mapping[str, Any] | None = None,
	write_csv: bool = True,
) -> dict[str, Any]:
	existing_path = baseline_results_dir / f"{method_name}_baseline_{split}.csv"
	loaded = _load_baseline_csv(existing_path, method_name, split, expected_sequence=expected_sequence)
	if loaded is not None:
		logger.info("Loaded existing %s baseline results from %s", method_name, existing_path)
		return loaded
	from src.baselines import evaluate_baseline, predict_linear_extrapolation_for_sample, predict_persistence_for_sample

	predict_fn = predict_persistence_for_sample if method_name == "persistence" else predict_linear_extrapolation_for_sample
	output_csv = output_dir / "logs" / f"{method_name}_baseline_{split}.csv" if write_csv else None
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
	paper_energy_metric: str = "log",
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
	aggregate = result.get("aggregate_results", {})
	if not isinstance(aggregate, Mapping):
		aggregate = {}
	values, sources = paper_values_from_metrics(aggregate, paper_energy_metric=paper_energy_metric)
	apply_skill(values, persistence_values, is_persistence=is_persistence, paper_energy_metric=paper_energy_metric)
	paper_row = build_paper_row(identity["model_name"], values, paper_energy_metric=paper_energy_metric)
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
		paper_energy_metric=paper_energy_metric,
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
		paper_energy_metric=paper_energy_metric,
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
		"normalization": dict(result.get("normalization", {})) if isinstance(result.get("normalization", {}), Mapping) else {},
		"paper_values": values,
		"metric_sources": sources,
		"secondary_energy_metrics": _secondary_energy_payload(aggregate),
		"num_samples": int(result.get("num_samples", 0)),
	}
	return paper_row, wide_row, per_fire, long_rows, metadata


def _secondary_energy_payload(metrics: Mapping[str, Any]) -> dict[str, Any]:
	energy_mw, _ = _metric(metrics, ["test_energy_mw_mae", "test_energy_MW_mae", "energy_mw_mae", "energy_MW_mae"])
	active_energy_mw, _ = _metric(
		metrics,
		["test_active_energy_mw_mae", "test_energy_mw_active_mae", "test_energy_MW_active_mae", "active_energy_mw_mae", "energy_MW_active_mae"],
	)
	energy_mw_rmse, _ = _metric(metrics, ["test_energy_mw_rmse", "test_energy_MW_rmse", "energy_mw_rmse", "energy_MW_rmse"])
	active_energy_mw_rmse, _ = _metric(
		metrics,
		["test_active_energy_mw_rmse", "test_energy_mw_active_rmse", "test_energy_MW_active_rmse", "active_energy_mw_rmse", "energy_MW_active_rmse"],
	)
	return {
		"energy_mw_mae": energy_mw,
		"active_energy_mw_mae": active_energy_mw,
		"energy_mw_rmse": energy_mw_rmse,
		"active_energy_mw_rmse": active_energy_mw_rmse,
	}


def _paper_metric_payload(values: Mapping[str, Any], paper_energy_metric: str = "log") -> dict[str, Any]:
	energy_column, active_energy_column = _energy_paper_columns(paper_energy_metric)
	return {
		"surf_mae": values.get("Surf. MAE ↓"),
		"canopy_mae": values.get("Canopy MAE ↓"),
		"dice": values.get("Dice ↑"),
		"iou": values.get("IoU ↑"),
		"energy_log_mae": values.get("Energy Log MAE ↓"),
		"active_energy_log_mae": values.get("Active Energy Log MAE ↓"),
		"energy_mw_mae": values.get("Energy MW MAE ↓"),
		"active_energy_mw_mae": values.get("Active Energy MW MAE ↓"),
		"paper_energy_metric": str(paper_energy_metric).lower(),
		"paper_energy_mae": values.get(energy_column),
		"paper_active_energy_mae": values.get(active_energy_column),
		"skill": values.get("Skill ↑"),
	}


def _format_report_float(value: Any) -> str:
	number = _safe_float(value)
	if not math.isfinite(number):
		return "NaN"
	return f"{number:.6g}"


def _markdown_metrics_table(rows: list[Mapping[str, Any]], paper_energy_metric: str = "log") -> list[str]:
	energy_column, active_energy_column = _energy_paper_columns(paper_energy_metric)
	lines = [
		f"| Model | Surf. MAE ↓ | Canopy MAE ↓ | Dice ↑ | IoU ↑ | {energy_column} | {active_energy_column} | Skill ↑ |",
		"|---|---:|---:|---:|---:|---:|---:|---:|",
	]
	for row in rows:
		lines.append(
			"| "
			+ " | ".join(
				[
					str(row.get("Model", "")),
					_format_report_float(row.get("Surf. MAE ↓")),
					_format_report_float(row.get("Canopy MAE ↓")),
					_format_report_float(row.get("Dice ↑")),
					_format_report_float(row.get("IoU ↑")),
					_format_report_float(row.get(energy_column)),
					_format_report_float(row.get(active_energy_column)),
					_format_report_float(row.get("Skill ↑")),
				]
			)
			+ " |"
		)
	return lines


def _markdown_per_fire_table(rows: list[Mapping[str, Any]], limit: int | None = None) -> list[str]:
	lines = [
		"| Model | Fire | Surf. MAE | Canopy MAE | Dice | IoU | Energy Log MAE | Active Energy Log MAE | Skill |",
		"|---|---|---:|---:|---:|---:|---:|---:|---:|",
	]
	visible_rows = rows if limit is None else rows[:limit]
	for row in visible_rows:
		lines.append(
			"| "
			+ " | ".join(
				[
					display_name(str(row.get("architecture", ""))),
					str(row.get("fire_name", "")),
					_format_report_float(row.get("surf_mae")),
					_format_report_float(row.get("canopy_mae")),
					_format_report_float(row.get("dice")),
					_format_report_float(row.get("iou")),
					_format_report_float(row.get("energy_log_mae")),
					_format_report_float(row.get("active_energy_log_mae")),
					_format_report_float(row.get("skill")),
				]
			)
			+ " |"
		)
	if limit is not None and len(rows) > limit:
		lines.append(f"\nShowing first {limit} of {len(rows)} per-fire rows. See JSON/CSV outputs for all rows.")
	return lines


def _build_evaluated_model_entries(
	json_entries: list[Mapping[str, Any]],
	per_fire_rows: list[Mapping[str, Any]],
	type_by_architecture: Mapping[str, str],
	paper_energy_metric: str = "log",
) -> list[dict[str, Any]]:
	entries: list[dict[str, Any]] = []
	for entry in json_entries:
		identity = entry.get("identity", {}) if isinstance(entry.get("identity"), Mapping) else {}
		architecture = str(identity.get("architecture", ""))
		paper_values = entry.get("paper_values", {}) if isinstance(entry.get("paper_values"), Mapping) else {}
		model_per_fire = [
			dict(row)
			for row in per_fire_rows
			if str(row.get("architecture", "")) == architecture
			and str(row.get("run_name", "")) == str(identity.get("run_name", ""))
		]
		secondary_metrics = entry.get("secondary_energy_metrics", {}) if isinstance(entry.get("secondary_energy_metrics"), Mapping) else {}
		metrics_payload = _paper_metric_payload(paper_values, paper_energy_metric=paper_energy_metric)
		metrics_payload["secondary"] = dict(secondary_metrics)
		for metric_name, metric_value in secondary_metrics.items():
			metrics_payload.setdefault(str(metric_name), metric_value)
		entries.append(
			{
				"architecture": architecture,
				"display_name": str(identity.get("model_name", display_name(architecture))),
				"type": type_by_architecture.get(architecture, "learned"),
				"checkpoint": identity.get("checkpoint_path"),
				"run_name": identity.get("run_name"),
				"metrics": metrics_payload,
				"secondary_metrics": dict(secondary_metrics),
				"sequence": entry.get("sequence", {}),
				"normalization": entry.get("normalization", {}),
				"per_fire_metrics": model_per_fire,
			}
		)
	return entries


def _write_reports(report_md_path: Path, report_json_path: Path | None, report: Mapping[str, Any], write_json: bool = False) -> None:
	if write_json and report_json_path is not None:
		_write_json(report_json_path, report)
	lines = [
		"# Evaluation Report",
		"",
		"## Command",
		"",
		f"`{report.get('command', '')}`",
		"",
		"## Evaluation Setup",
		"",
	]
	setup = report.get("setup", {}) if isinstance(report.get("setup"), Mapping) else {}
	for key in ("timestamp", "split", "mode", "requested_model_architecture", "resolved_evaluated_models", "config", "runs_root", "output_dir", "device", "max_batches"):
		lines.append(f"- {key}: {setup.get(key)}")
	lines.extend(["", "## Sequence / Horizon", ""])
	sequence = report.get("sequence", {}) if isinstance(report.get("sequence"), Mapping) else {}
	for key, value in sequence.items():
		lines.append(f"- {key}: {value}")
	if sequence.get("input_sequence_length") and sequence.get("prediction_horizon"):
		lines.append(
			f"- statement: This evaluation uses {sequence.get('input_sequence_length')} input frames "
			f"and predicts {sequence.get('prediction_horizon')} steps after the last input frame."
		)
	lines.extend(["", "## Normalization", ""])
	normalization = report.get("normalization", {}) if isinstance(report.get("normalization"), Mapping) else {}
	for key, value in normalization.items():
		lines.append(f"- {key}: {_to_jsonable(value)}")
	lines.extend(["", "## Cache", ""])
	cache = report.get("cache", {}) if isinstance(report.get("cache"), Mapping) else {}
	for key, value in cache.items():
		lines.append(f"- {key}: {_to_jsonable(value)}")
	lines.extend(["", "## Selected Checkpoints", ""])
	selected = report.get("selected_checkpoints", [])
	if selected:
		for item in selected:
			lines.append(f"### {item.get('architecture')}")
			for key, value in item.items():
				lines.append(f"- {key}: {_to_jsonable(value)}")
			lines.append("")
	else:
		lines.append("No learned checkpoints selected.")
	lines.extend(["", "## Paper Metrics Summary", ""])
	paper_energy_metric = str(setup.get("paper_energy_metric", "log"))
	if paper_energy_metric == "log":
		lines.append(
			"Energy metrics in the main paper table are computed in log-space on channel 3: "
			"MAE(pred_log1p_energy, target_log1p_energy). MW-space energy metrics are reported as secondary diagnostics but are not used for Skill."
		)
		lines.append("")
	lines.extend(_markdown_metrics_table(list(report.get("paper_rows", [])), paper_energy_metric=paper_energy_metric))
	if paper_energy_metric == "log":
		lines.extend(["", "## Secondary MW Energy Diagnostics", ""])
		for model in report.get("evaluated_models", []):
			if not isinstance(model, Mapping):
				continue
			metrics = model.get("metrics", {}) if isinstance(model.get("metrics"), Mapping) else {}
			secondary = metrics.get("secondary", {}) if isinstance(metrics.get("secondary"), Mapping) else {}
			lines.append(
				f"- {model.get('display_name')}: Energy MW MAE={_format_report_float(secondary.get('energy_mw_mae', metrics.get('energy_mw_mae')))}; "
				f"Active Energy MW MAE={_format_report_float(secondary.get('active_energy_mw_mae', metrics.get('active_energy_mw_mae')))}"
			)
	lines.extend(["", "## Per-Fire Summary", ""])
	lines.extend(_markdown_per_fire_table(list(report.get("per_fire_rows", []))))
	lines.extend(["", "## Warnings", ""])
	warnings = report.get("warnings", [])
	if warnings:
		lines.extend(f"- {warning}" for warning in warnings)
	else:
		lines.append("No warnings.")
	lines.extend(["", "## Failures", ""])
	failures = report.get("failures", [])
	if failures:
		for failure in failures:
			lines.append(f"- {failure}")
	else:
		lines.append("No failures.")
	lines.extend(["", "## Output Files", ""])
	output_files = report.get("output_files", {}) if isinstance(report.get("output_files"), Mapping) else {}
	if output_files:
		for key, value in output_files.items():
			lines.append(f"- {key}: {value}")
	else:
		lines.append("No sidecar output files were written.")
	report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_qualitative_output_dir(output_root: Path, split: str, model_architecture: str, overwrite: bool, eval_name: str = "auto") -> Path:
	return _make_output_dir(output_root / "qualitative", "qualitative", split, model_architecture, overwrite, eval_name=eval_name)


def _build_qualitative_loader_context(
	config_path: str | Path,
	split: str,
	config_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
	from src.data.dataset import create_dataloaders
	from src.models.evaluation import _deep_merge, _select_loader

	config = _load_config_with_path(config_path)
	if isinstance(config_override, Mapping):
		config = _deep_merge(config, dict(config_override))
	config["return_metadata"] = True
	train_loader, val_loader, test_loader = create_dataloaders(config)
	selected_loader = _select_loader(train_loader, val_loader, test_loader, split)
	if len(selected_loader.dataset) == 0:
		raise ValueError(f"Selected split {split!r} is empty; qualitative evaluation needs at least one sample.")
	return {
		"config": config,
		"train_loader": train_loader,
		"selected_loader": selected_loader,
		"dataset": selected_loader.dataset,
	}


def _selected_model_entry(
	*,
	key: str,
	model_type: str,
	status: str,
	run_name: str | None = None,
	run_dir: str | None = None,
	checkpoint: str | None = None,
	resolved_config_path: str | None = None,
	selection_source: str | None = None,
	error: str | None = None,
	warning: str | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
	entry: dict[str, Any] = {
		"key": str(key),
		"architecture": str(key),
		"display": display_name(key),
		"display_name": display_name(key),
		"type": str(model_type),
		"checkpoint": checkpoint,
		"run_name": run_name,
		"run": run_dir,
		"status": str(status),
	}
	if resolved_config_path is not None:
		entry["resolved_config_path"] = resolved_config_path
	if selection_source is not None:
		entry["selection_source"] = selection_source
	if error is not None:
		entry["error"] = error
	if warning is not None:
		entry["warning"] = warning
	if isinstance(metadata, Mapping):
		entry.update(dict(metadata))
	return entry


def _resolve_qualitative_learned_runs(
	*,
	args: argparse.Namespace,
	learned_targets: list[str],
	all_model_mode: bool,
	runs_root: Path,
	logger: logging.Logger,
	warnings_list: list[str],
	skipped_models: list[str],
	selected_models: list[dict[str, Any]],
) -> list[tuple[str, TrainingRun, str]]:
	selected: list[tuple[str, TrainingRun, str]] = []
	all_runs: list[TrainingRun] = []
	if learned_targets and args.checkpoint in (None, "", "null"):
		logger.info("Discovering runs under %s", runs_root)
		all_runs = discover_runs(runs_root, checkpoint_name=args.checkpoint_name)
	for requested_architecture in learned_targets:
		architecture = canonical_architecture(requested_architecture)
		if args.checkpoint not in (None, "", "null"):
			best_run = _explicit_training_run(architecture, args.checkpoint)
			selection_source = "explicit"
		else:
			runs = [run for run in all_runs if run.architecture.lower() == architecture]
			best_run = find_best_run(runs, architecture, selection_metric=args.selection_metric, selection_mode=args.selection_mode)
			selection_source = "auto"
		if best_run is None:
			message = f"No trained runs found for {requested_architecture}; skipping."
			if all_model_mode:
				logger.info(message)
				warnings_list.append(message)
				skipped_models.append(requested_architecture)
				selected_models.append(
					_selected_model_entry(
						key=requested_architecture,
						model_type="learned",
						status="skipped",
						warning=message,
						selection_source=selection_source,
					)
				)
				continue
			raise FileNotFoundError(f"No trained runs found for requested architecture {requested_architecture!r} under {runs_root}.")
		selected.append((architecture, best_run, selection_source))
	return selected


def _run_qualitative_baseline_predictions(
	*,
	method_name: str,
	config: Mapping[str, Any],
	samples: list[Mapping[str, Any]],
) -> list[Any]:
	from src.baselines.common import resolve_patch
	from src.baselines.evaluator import _prepare_dataset_records, _resolve_dataset_record
	from src.baselines.linear_extrapolation import predict_linear_extrapolation_for_sample
	from src.baselines.persistence import predict_persistence_for_sample
	from src.data.discovery import discover_multiple_datasets

	predict_fn = predict_persistence_for_sample if method_name == "persistence" else predict_linear_extrapolation_for_sample
	dataset_records = discover_multiple_datasets(config)
	prepared_records = _prepare_dataset_records(dataset_records, config)
	predictions: list[Any] = []
	for sample in samples:
		metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), Mapping) else {}
		metadata = dict(metadata)
		if "sample_index" not in metadata and "start_idx" in metadata:
			metadata["sample_index"] = metadata["start_idx"]
		dataset_record = _resolve_dataset_record(prepared_records, metadata)
		prediction = predict_fn(
			dataset_record=dataset_record,
			sample_ref=metadata,
			config=config,
			patch=resolve_patch(metadata=metadata),
		)
		predictions.append(coerce_chw4(prediction))
	return predictions


def _run_qualitative_learned_predictions(
	*,
	args: argparse.Namespace,
	architecture: str,
	best_run: TrainingRun,
	sample_indices: list[int],
) -> tuple[list[Any], dict[str, Any]]:
	try:
		import torch  # type: ignore[import-not-found]
	except ImportError as exc:  # pragma: no cover - environment-specific
		raise ImportError("PyTorch is required for qualitative learned-model evaluation.") from exc

	from src.models.evaluation import (
		_validate_checkpoint_architecture,
		_validate_checkpoint_sequence,
	)
	from src.models.model_factory import build_model_from_config
	from src.training.checkpoints import load_model_state_dict_compatible, validate_checkpoint_model_compatibility
	from src.training.hardware import autocast_context, choose_amp_dtype
	from src.training.input_normalization import (
		apply_input_normalization,
		build_input_normalizer_for_loader,
		compare_normalization_metadata,
		normalization_metadata_from_loader,
	)
	from src.training.train import _get_device, _infer_input_channels_from_loader

	config_path = best_run.resolved_config_path or args.config
	context = _build_qualitative_loader_context(
		config_path,
		args.split,
		config_override=_model_config_override(architecture, best_run.checkpoint_path, args.num_workers, run_dir=best_run.run_dir),
	)
	config = context["config"]
	train_loader = context["train_loader"]
	selected_loader = context["selected_loader"]
	dataset = context["dataset"]
	input_channels = _infer_input_channels_from_loader(train_loader)
	device = _get_device(config)
	model = build_model_from_config(config, input_channels=input_channels).to(device)
	input_normalizer = build_input_normalizer_for_loader(selected_loader, device, input_channels, config)
	normalization_metadata = normalization_metadata_from_loader(selected_loader, config, input_channels)
	checkpoint_path = Path(best_run.checkpoint_path).expanduser().resolve()
	checkpoint = load_checkpoint(checkpoint_path, map_location=device)
	_validate_checkpoint_architecture(checkpoint, architecture, checkpoint_path)
	_validate_checkpoint_sequence(
		checkpoint,
		config,
		checkpoint_path,
		allow_sequence_mismatch=bool(args.allow_sequence_mismatch),
	)
	normalization_mismatches = compare_normalization_metadata(checkpoint.get("normalization"), normalization_metadata)
	if normalization_mismatches:
		message = (
			f"Checkpoint normalization metadata mismatch for {checkpoint_path}:\n"
			+ "\n".join(f"  - {item}" for item in normalization_mismatches)
		)
		if bool(args.allow_normalization_mismatch):
			warnings.warn(message, RuntimeWarning, stacklevel=2)
		else:
			raise ValueError(message + "\nPass --allow_normalization_mismatch only for intentional compatibility/debug runs.")
	validate_checkpoint_model_compatibility(model, checkpoint, checkpoint_path)
	load_model_state_dict_compatible(model, checkpoint, checkpoint_path)
	model.eval()
	amp_dtype = choose_amp_dtype(config, device)

	predictions: list[Any] = []
	with torch.inference_mode():
		for dataset_index in sample_indices:
			item = dataset[int(dataset_index)]
			if isinstance(item, dict):
				x = item["x"]; target = coerce_chw4(item["y"]); terrain = item.get("terrain")
			elif isinstance(item, (tuple, list)) and len(item) >= 2:
				x = item[0]; target = coerce_chw4(item[1]); terrain = None
			else:
				raise TypeError("Qualitative dataset items must contain at least input and target tensors.")
			if not torch.is_tensor(x):
				x = torch.as_tensor(x, dtype=torch.float32)
			if x.ndim == 4:
				x_batch = x.unsqueeze(0)
			elif x.ndim == 5 and int(x.shape[0]) == 1:
				x_batch = x
			else:
				raise ValueError(f"Expected input sample shape (T, C, H, W), got {tuple(x.shape)}.")
			x_batch = x_batch.to(device, non_blocking=True)
			terrain_batch = terrain.to(device, non_blocking=True) if terrain is not None else None
			if terrain_batch is not None and terrain_batch.ndim == 3:
				terrain_batch = terrain_batch.unsqueeze(0)
			x_batch = apply_input_normalization(x_batch, input_normalizer, config)
			with autocast_context(device, amp_dtype):
				output = model(x_batch, terrain=terrain_batch) if terrain_batch is not None else model(x_batch)
			output = extract_prediction(output)
			if not torch.is_tensor(output):
				raise TypeError(f"Model output must be a torch Tensor, got {type(output)!r}.")
			prediction = coerce_chw4(output.detach().float().cpu())
			if tuple(prediction.shape[-2:]) != tuple(target.shape[-2:]):
				raise ValueError(
					f"Prediction spatial shape {tuple(prediction.shape[-2:])} does not match target {tuple(target.shape[-2:])}."
				)
			predictions.append(prediction)

	return predictions, {
		"device": str(device),
		"sequence": _sequence_metadata(config),
		"normalization": normalization_metadata,
	}


def _write_qualitative_reports(report_md_path: Path, report_json_path: Path, report: Mapping[str, Any]) -> None:
	_write_json(report_json_path, report)
	lines = [
		"# Qualitative Evaluation Report",
		"",
		"## Command",
		"",
		f"`{report.get('command', '')}`",
		"",
		"## Evaluation Setup",
		"",
	]
	setup = report.get("setup", {}) if isinstance(report.get("setup"), Mapping) else {}
	for key in (
		"timestamp",
		"config",
		"split",
		"mode",
		"requested_model_architecture",
		"num_samples_requested",
		"num_samples_selected",
		"qualitative_seed",
		"qualitative_output_format",
		"qualitative_dpi",
		"output_dir",
		"image_dir",
	):
		lines.append(f"- {key}: {setup.get(key)}")
	lines.extend(["", "## Sequence / Horizon", ""])
	sequence = report.get("sequence", {}) if isinstance(report.get("sequence"), Mapping) else {}
	for key, value in sequence.items():
		lines.append(f"- {key}: {_to_jsonable(value)}")
	lines.append("- rollout: one-shot predictions only; recursive rollout is not part of this qualitative mode.")
	lines.extend(["", "## Normalization", ""])
	normalization = report.get("normalization", {}) if isinstance(report.get("normalization"), Mapping) else {}
	for key, value in normalization.items():
		lines.append(f"- {key}: {_to_jsonable(value)}")
	lines.extend(["", "## Baseline Settings", ""])
	baselines = report.get("baseline_settings", {}) if isinstance(report.get("baseline_settings"), Mapping) else {}
	if baselines:
		for key, value in baselines.items():
			lines.append(f"- {key}: {_to_jsonable(value)}")
	else:
		lines.append("No baseline settings found in config.")
	lines.extend(["", "## Evaluated Models", ""])
	evaluated_models = report.get("evaluated_models", [])
	if evaluated_models:
		for model in evaluated_models:
			lines.append(f"- {model.get('display_name')}: type={model.get('type')}; checkpoint={model.get('checkpoint')}; run={model.get('run_name')}")
	else:
		lines.append("No models were evaluated.")
	lines.extend(["", "## Selected Samples", ""])
	selected_samples = report.get("selected_samples", [])
	if selected_samples:
		lines.append(", ".join(str(sample.get("dataset_index")) for sample in selected_samples))
	else:
		lines.append("No samples selected.")
	lines.extend(["", "## Images", ""])
	for image in report.get("image_paths", []):
		lines.append(f"- {image}")
	lines.extend(["", "## Warnings", ""])
	warnings_list = report.get("warnings", [])
	if warnings_list:
		lines.extend(f"- {warning}" for warning in warnings_list)
	else:
		lines.append("No warnings.")
	lines.extend(["", "## Failures", ""])
	failures = report.get("failures", [])
	if failures:
		for failure in failures:
			lines.append(f"- {failure}")
	else:
		lines.append("No failures.")
	report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_qualitative_mode(args: argparse.Namespace | None = None) -> Path:
	if args is None:
		raise ValueError("run_qualitative_mode requires parsed CLI arguments.")
	output_root = Path(args.output_dir).expanduser().resolve()
	output_dir = _make_qualitative_output_dir(output_root, args.split, args.model_architecture, bool(args.overwrite), eval_name=args.eval_name)
	logger = _setup_logger(output_dir, write_log_file=bool(args.write_log_file))

	learned_targets, baseline_targets, all_model_mode = _resolve_requested_targets(args.model_architecture, include_baselines=bool(args.include_baselines))
	_validate_checkpoint_arg(args.checkpoint, learned_targets, baseline_targets, args.model_architecture)
	runs_root = Path(args.runs_root).expanduser().resolve()
	warnings_list: list[str] = []
	failures: list[dict[str, Any]] = []
	skipped_models: list[str] = []
	successful_models: list[str] = []
	selected_models: list[dict[str, Any]] = []
	selected_learned = _resolve_qualitative_learned_runs(
		args=args,
		learned_targets=learned_targets,
		all_model_mode=all_model_mode,
		runs_root=runs_root,
		logger=logger,
		warnings_list=warnings_list,
		skipped_models=skipped_models,
		selected_models=selected_models,
	)
	sample_config_path = args.config
	sample_config_override: Mapping[str, Any] = _num_worker_override(args.num_workers)
	if not baseline_targets and len(selected_learned) == 1:
		selected_architecture, selected_run, _selection_source = selected_learned[0]
		if selected_run.resolved_config_path not in (None, "", "null"):
			sample_config_path = selected_run.resolved_config_path
			sample_config_override = _model_config_override(
				selected_architecture,
				selected_run.checkpoint_path,
				args.num_workers,
				run_dir=selected_run.run_dir,
			)
			logger.info(
				"Selecting qualitative samples from checkpoint resolved config %s instead of CLI base config %s.",
				sample_config_path,
				args.config,
			)
	base_context = _build_qualitative_loader_context(sample_config_path, args.split, config_override=sample_config_override)
	base_config = base_context["config"]
	sequence_setup = _sequence_metadata(base_config)
	normalization_setup = _normalization_setup_metadata(base_config)
	baseline_settings = dict(base_config.get("baselines", {})) if isinstance(base_config.get("baselines"), Mapping) else {}
	dataset = base_context["dataset"]
	dataset_length = len(dataset)
	if args.max_batches is not None:
		warnings_list.append("--max_batches is ignored in qualitative mode; use --num_samples to control image count.")
	if int(args.num_samples) > dataset_length:
		warnings_list.append(
			f"Requested num_samples={int(args.num_samples)} exceeds split length={dataset_length}; using {dataset_length} sample(s)."
		)
	selected_indices = select_qualitative_sample_indices(dataset_length, int(args.num_samples), int(args.qualitative_seed))
	samples = load_qualitative_samples(
		dataset,
		selected_indices,
		split=args.split,
		input_sequence_length=int(sequence_setup["input_sequence_length"]),
		prediction_horizon=int(sequence_setup["prediction_horizon"]),
	)
	predictions_by_sample: list[dict[str, Any]] = [{} for _ in samples]
	logger.info("Qualitative output directory: %s", output_dir)
	logger.info("Selected qualitative sample indices: %s", selected_indices)

	def _remove_model_predictions(model_key: str) -> None:
		for sample_predictions in predictions_by_sample:
			sample_predictions.pop(model_key, None)

	def _record_failure(model_key: str, exc: Exception, model_type: str = "learned", extra: Mapping[str, Any] | None = None) -> None:
		_remove_model_predictions(model_key)
		failure = {"architecture": model_key, "type": model_type, "error": str(exc)}
		if isinstance(extra, Mapping):
			failure.update(dict(extra))
		failures.append(failure)
		selected_models.append(
			_selected_model_entry(
				key=model_key,
				model_type=model_type,
				status="failed",
				error=str(exc),
				metadata=extra,
			)
		)

	for method_name in baseline_targets:
		logger.info("Running qualitative baseline predictions for %s", method_name)
		try:
			baseline_predictions = _run_qualitative_baseline_predictions(method_name=method_name, config=base_config, samples=samples)
			for sample_number, prediction in enumerate(baseline_predictions):
				predictions_by_sample[sample_number][method_name] = prediction
			selected_models.append(_selected_model_entry(key=method_name, model_type="baseline", status="completed"))
			successful_models.append(method_name)
		except Exception as exc:
			logger.exception("Qualitative baseline %s failed", method_name)
			_record_failure(method_name, exc, model_type="baseline")
			if not all_model_mode:
				raise
			warnings_list.append(f"Baseline {method_name} failed: {exc}")

	for architecture, best_run, selection_source in selected_learned:
		logger.info("Running qualitative learned predictions for %s from %s", architecture, best_run.checkpoint_path)
		metadata = {
			"run_name": best_run.run_name,
			"run": best_run.run_dir,
			"checkpoint": best_run.checkpoint_path,
			"resolved_config_path": best_run.resolved_config_path,
			"selection_source": selection_source,
		}
		try:
			with warnings.catch_warnings(record=True) as caught_warnings:
				warnings.simplefilter("always", RuntimeWarning)
				model_predictions, model_metadata = _run_qualitative_learned_predictions(
					args=args,
					architecture=architecture,
					best_run=best_run,
					sample_indices=selected_indices,
				)
			for caught in caught_warnings:
				warnings_list.append(str(caught.message))
			for sample_number, prediction in enumerate(model_predictions):
				predictions_by_sample[sample_number][architecture] = prediction
			selected_models.append(
				_selected_model_entry(
					key=architecture,
					model_type="learned",
					status="completed",
					run_name=best_run.run_name,
					run_dir=best_run.run_dir,
					checkpoint=best_run.checkpoint_path,
					resolved_config_path=best_run.resolved_config_path,
					selection_source=selection_source,
					metadata=model_metadata,
				)
			)
			successful_models.append(architecture)
		except Exception as exc:
			logger.exception("Qualitative learned evaluation failed for %s", architecture)
			_record_failure(architecture, exc, model_type="learned", extra=metadata)
			if not all_model_mode:
				raise
			warnings_list.append(f"Evaluation failed for {architecture}: {exc}")

	evaluated_models = [model for model in selected_models if str(model.get("status")) == "completed"]
	if not evaluated_models:
		raise RuntimeError("Qualitative evaluation has no successful models to plot.")

	image_dir = output_dir / "images"
	if bool(args.overwrite) and image_dir.exists():
		for old_image in image_dir.glob("sample_*"):
			if old_image.is_file():
				old_image.unlink()
	image_paths: list[str] = []
	individual_image_paths: list[str] = []
	output_format = str(args.qualitative_output_format).lower()
	for sample in samples:
		sample_number = int(sample["sample_number"])
		image_path = image_dir / f"sample_{sample_number:03d}.{output_format}"
		save_qualitative_summary_image(
			target=sample["target"],
			predictions=predictions_by_sample[sample_number],
			models=evaluated_models,
			sample_record=sample["record"],
			output_path=image_path,
			dpi=int(args.qualitative_dpi),
		)
		image_paths.append(str(image_path))
		if bool(args.save_individual_model_panels):
			paths = save_individual_model_images(
				target=sample["target"],
				predictions=predictions_by_sample[sample_number],
				models=evaluated_models,
				sample_record=sample["record"],
				output_root=image_dir / "individual",
				output_format=output_format,
				dpi=int(args.qualitative_dpi),
			)
			individual_image_paths.extend(str(path) for path in paths)

	selected_sample_records = [dict(sample["record"]) for sample in samples]
	_write_json(output_dir / "selected_samples.json", selected_sample_records)
	_write_json(output_dir / "selected_models.json", selected_models)
	report = {
		"command": " ".join(sys.argv),
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"setup": {
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"config": str(Path(args.config).expanduser().resolve()),
			"split": args.split,
			"mode": "qualitative",
			"requested_model_architecture": args.model_architecture,
			"num_samples_requested": int(args.num_samples),
			"num_samples_selected": len(samples),
			"qualitative_seed": int(args.qualitative_seed),
			"qualitative_output_format": output_format,
			"qualitative_dpi": int(args.qualitative_dpi),
			"output_dir": str(output_dir),
			"image_dir": str(image_dir),
			"runs_root": str(runs_root),
			"checkpoint": args.checkpoint,
		},
		"sequence": sequence_setup,
		"normalization": normalization_setup,
		"baseline_settings": baseline_settings,
		"selected_models": selected_models,
		"evaluated_models": evaluated_models,
		"selected_samples": selected_sample_records,
		"selected_sample_indices": selected_indices,
		"image_paths": image_paths,
		"individual_image_paths": individual_image_paths,
		"warnings": warnings_list,
		"failures": failures,
		"skipped_models": skipped_models,
		"successful_models": successful_models,
		"note": "Qualitative mode is one-shot: each selected sample is predicted independently with no recursive rollout.",
	}
	_write_qualitative_reports(output_dir / "qualitative_report.md", output_dir / "qualitative_report.json", report)
	print(f"successful models: {', '.join(successful_models) if successful_models else 'none'}")
	print(f"skipped models: {', '.join(skipped_models) if skipped_models else 'none'}")
	print(f"failed models: {', '.join(item['architecture'] for item in failures) if failures else 'none'}")
	print(f"report: {output_dir / 'qualitative_report.md'}")
	print(f"output_dir: {output_dir}")
	return output_dir


def run_quantitative(args: argparse.Namespace) -> Path:
	output_root = Path(args.output_dir).expanduser().resolve()
	output_dir = _make_output_dir(output_root, "quantitative", args.split, args.model_architecture, bool(args.overwrite), eval_name=args.eval_name)
	logger = _setup_logger(output_dir, write_log_file=bool(args.write_log_file))
	if args.save_predictions:
		logger.warning("Model prediction saving is not implemented in the shared quantitative checkpoint evaluator yet.")

	learned_targets, baseline_targets, all_model_mode = _resolve_requested_targets(args.model_architecture, include_baselines=bool(args.include_baselines))
	_validate_checkpoint_arg(args.checkpoint, learned_targets, baseline_targets, args.model_architecture)
	runs_root = Path(args.runs_root).expanduser().resolve()
	base_config = _load_config_with_path(args.config)
	sequence_setup = _sequence_metadata(base_config)
	normalization_setup = _normalization_setup_metadata(base_config)
	cache_setup: dict[str, Any] = {"checked": False}
	selected_runs: list[TrainingRun] = []
	selected_checkpoint_infos: list[dict[str, Any]] = []
	selected_rows: list[dict[str, Any]] = []
	skipped_models: list[str] = []
	failed_models: list[dict[str, str]] = []
	successful_models: list[str] = []
	warnings_list: list[str] = []
	paper_rows: list[dict[str, Any]] = []
	wide_rows: list[dict[str, Any]] = []
	per_fire_rows: list[dict[str, Any]] = []
	long_rows: list[dict[str, Any]] = []
	json_entries: list[dict[str, Any]] = []
	type_by_architecture: dict[str, str] = {}
	persistence_values: dict[str, float] | None = None
	persistence_by_fire: dict[str, dict[str, float]] | None = None

	logger.info("Evaluation output directory: %s", output_dir)
	logger.info("Requested learned targets: %s", learned_targets)
	logger.info("Requested baseline targets: %s", baseline_targets)
	logger.info(
		"Sequence configuration | input_sequence_length=%s prediction_horizon=%s target_offset_from_start=%s split=%s cache_version=%s dataset_index=%s",
		sequence_setup.get("input_sequence_length"),
		sequence_setup.get("prediction_horizon"),
		sequence_setup.get("target_offset_from_start"),
		args.split,
		sequence_setup.get("cache_version"),
		sequence_setup.get("dataset_index"),
	)

	def _ensure_cache_checked() -> None:
		nonlocal cache_setup
		if cache_setup.get("checked"):
			return
		cache_setup = {"checked": True, **_cache_metadata(base_config, args.split)}
		logger.info("Cache metadata: %s", cache_setup)

	def _run_baseline(method_name: str, include_row: bool) -> dict[str, Any] | None:
		nonlocal persistence_values, persistence_by_fire
		try:
			_ensure_cache_checked()
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
				expected_sequence=sequence_setup,
				write_csv=bool(args.include_csv_outputs),
			)
			logger.info("Baseline %s sequence: %s", method_name, result.get("sequence", {}))
			if method_name == "persistence":
				persistence_values, _sources = paper_values_from_metrics(result.get("aggregate_results", {}), paper_energy_metric=args.paper_energy_metric)
				apply_skill(persistence_values, None, is_persistence=True, paper_energy_metric=args.paper_energy_metric)
				persistence_by_fire = {}
			identity = _result_identity(method_name, method_name, "")
			paper_row, wide_row, baseline_per_fire, baseline_long, metadata = _result_rows(
				identity=identity,
				result=result,
				split=args.split,
				status="completed",
				persistence_values=persistence_values,
				persistence_by_fire=persistence_by_fire,
				is_persistence=method_name == "persistence",
				paper_energy_metric=args.paper_energy_metric,
			)
			if method_name == "persistence":
				persistence_by_fire = {}
				for row in baseline_per_fire:
					persistence_by_fire[str(row["fire_name"])] = {
						"Active Energy Log MAE ↓": _safe_float(row.get("active_energy_log_mae")),
						"Energy Log MAE ↓": _safe_float(row.get("energy_log_mae")),
						"Active Energy MW MAE ↓": _safe_float(row.get("active_energy_mw_mae")),
						"Energy MW MAE ↓": _safe_float(row.get("energy_mw_mae")),
					}
			if include_row:
				paper_rows.append(paper_row)
				wide_rows.append(wide_row)
				per_fire_rows.extend(baseline_per_fire)
				long_rows.extend(baseline_long)
				json_entries.append(metadata)
				type_by_architecture[method_name] = "baseline"
				successful_models.append(method_name)
			return result
		except Exception as exc:
			logger.exception("Baseline %s failed", method_name)
			failed_models.append({"architecture": method_name, "error": str(exc)})
			if include_row and not all_model_mode:
				raise
			warnings_list.append(f"Baseline {method_name} failed: {exc}")
			return None

	for method_name in baseline_targets:
		_run_baseline(method_name, include_row=True)

	all_runs: list[TrainingRun] = []
	if learned_targets and args.checkpoint in (None, "", "null"):
		logger.info("Discovering runs under %s", runs_root)
		all_runs = discover_runs(runs_root, checkpoint_name=args.checkpoint_name)

	for requested_architecture in learned_targets:
		architecture = canonical_architecture(requested_architecture)
		if args.checkpoint not in (None, "", "null"):
			best_run = _explicit_training_run(architecture, args.checkpoint)
			selected_value, selected_metric = None, "explicit_checkpoint"
			selection_source = "explicit"
		else:
			runs = [run for run in all_runs if run.architecture.lower() == architecture]
			best_run = find_best_run(runs, architecture, selection_metric=args.selection_metric, selection_mode=args.selection_mode)
			selected_value, selected_metric = (None, None) if best_run is None else metric_value_for_run(best_run, args.selection_metric)
			selection_source = "auto"
		if best_run is None:
			message = f"No trained runs found for {requested_architecture}; skipping."
			if all_model_mode:
				logger.info(message)
				print(message)
				skipped_models.append(requested_architecture)
				warnings_list.append(message)
				continue
			raise FileNotFoundError(f"No trained runs found for requested architecture {requested_architecture!r} under {runs_root}.")
		selected_runs.append(best_run)
		checkpoint_info = {
			"architecture": architecture,
			"run_name": best_run.run_name,
			"checkpoint_path": best_run.checkpoint_path,
			"checkpoint": _checkpoint_metadata(best_run.checkpoint_path),
			"best_epoch": best_run.best_epoch,
			"best_metric_name": best_run.best_metric_name,
			"best_metric_value": best_run.best_metric_value,
			"run_status": best_run.status,
			"selection_source": selection_source,
			"resolved_config_path": best_run.resolved_config_path,
		}
		selected_checkpoint_infos.append(checkpoint_info)
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
				"selection_source": selection_source,
			}
		)
		logger.info(
			"Selected %s | run=%s | metric=%s | value=%s | checkpoint=%s | source=%s",
			architecture,
			best_run.run_name,
			selected_metric,
			selected_value,
			best_run.checkpoint_path,
			selection_source,
		)
		try:
			if persistence_values is None and "persistence" not in baseline_targets:
				denominator_result = _run_baseline("persistence", include_row=False)
				if denominator_result is None:
					warnings_list.append("Persistence denominator for skill is unavailable; learned-model skill values may be NaN.")
				else:
					energy_column, active_energy_column = _energy_paper_columns(args.paper_energy_metric)
					active_denominator = None if persistence_values is None else persistence_values.get(active_energy_column)
					energy_denominator = None if persistence_values is None else persistence_values.get(energy_column)
					warnings_list.append(
						"Persistence denominator used for skill without adding a baseline row: "
						f"metric={active_energy_column}; active_value={active_denominator}; fallback_metric={energy_column}; fallback_value={energy_denominator}."
					)
			_ensure_cache_checked()
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
				allow_normalization_mismatch=bool(args.allow_normalization_mismatch),
			)
			identity = _result_identity(architecture, best_run.run_name, best_run.checkpoint_path)
			paper_row, wide_row, model_per_fire, model_long, metadata = _result_rows(
				identity=identity,
				result=result,
				split=args.split,
				status="completed",
				persistence_values=persistence_values,
				persistence_by_fire=persistence_by_fire,
				paper_energy_metric=args.paper_energy_metric,
			)
			paper_rows.append(paper_row)
			wide_rows.append(wide_row)
			per_fire_rows.extend(model_per_fire)
			long_rows.extend(model_long)
			json_entries.append(metadata)
			type_by_architecture[architecture] = "learned"
			successful_models.append(architecture)
		except Exception as exc:
			logger.exception("Evaluation failed for %s", architecture)
			failed_models.append({"architecture": architecture, "run_name": best_run.run_name, "error": str(exc)})
			if not all_model_mode:
				raise
			warnings_list.append(f"Evaluation failed for {architecture}: {exc}")

	if bool(args.copy_selected_run_metadata):
		_copy_selected_run_metadata(selected_runs, output_dir)

	output_files: dict[str, str] = {
		"evaluation_report_md": str(output_dir / "evaluation_report.md"),
	}
	if bool(args.include_json_outputs):
		output_files.update(
			{
				"evaluation_report_json": str(output_dir / "evaluation_report.json"),
				"evaluation_summary_json": str(output_dir / "evaluation_summary.json"),
			}
		)
	if bool(args.copy_selected_run_metadata) and selected_runs:
		output_files["selected_run_summaries_dir"] = str(output_dir / "selected_run_summaries")
	if bool(args.include_csv_outputs):
		_write_csv(output_dir / "selected_runs.csv", selected_rows)
		_write_json(output_dir / "selected_runs.json", selected_rows)
		_write_csv(output_dir / "paper_metrics.csv", paper_rows, paper_columns_for_metric(args.paper_energy_metric))
		_write_csv(output_dir / "paper_metrics_wide.csv", wide_rows)
		_write_json(output_dir / "paper_metrics.json", json_entries)
		_write_csv(output_dir / "per_fire_metrics.csv", per_fire_rows, PER_FIRE_COLUMNS)
		_write_csv(output_dir / "all_metrics_long.csv", long_rows)
		output_files.update(
			{
				"selected_runs_csv": str(output_dir / "selected_runs.csv"),
				"paper_metrics_csv": str(output_dir / "paper_metrics.csv"),
				"paper_metrics_wide_csv": str(output_dir / "paper_metrics_wide.csv"),
				"per_fire_metrics_csv": str(output_dir / "per_fire_metrics.csv"),
				"all_metrics_long_csv": str(output_dir / "all_metrics_long.csv"),
			}
		)
	if bool(args.write_latex):
		table_text = render_latex_table(paper_rows, bold_best=bool(args.latex_bold_best), paper_energy_metric=args.paper_energy_metric)
		(output_dir / "paper_table.tex").write_text(table_text, encoding="utf-8")
		output_files["paper_table_tex"] = str(output_dir / "paper_table.tex")
	if bool(args.write_log_file):
		output_files["log_file"] = str(output_dir / "logs" / "evaluate_trained_models.log")

	resolved_models = [entry.get("identity", {}).get("architecture") for entry in json_entries if isinstance(entry.get("identity"), Mapping)]
	evaluated_models = _build_evaluated_model_entries(json_entries, per_fire_rows, type_by_architecture, paper_energy_metric=args.paper_energy_metric)
	command = " ".join(sys.argv)
	report = {
		"command": command,
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"setup": {
			"timestamp": datetime.now(timezone.utc).isoformat(),
			"split": args.split,
			"mode": "quantitative",
			"requested_model_architecture": args.model_architecture,
			"resolved_evaluated_models": resolved_models,
			"config": str(Path(args.config).expanduser().resolve()),
			"runs_root": str(runs_root),
			"output_dir": str(output_dir),
			"device": base_config.get("training", {}).get("device", base_config.get("device", "auto")) if isinstance(base_config.get("training"), Mapping) else base_config.get("device", "auto"),
			"max_batches": args.max_batches,
			"checkpoint": args.checkpoint,
			"paper_energy_metric": args.paper_energy_metric,
		},
		"sequence": sequence_setup,
		"cache": cache_setup,
		"normalization": normalization_setup,
		"selected_checkpoints": selected_checkpoint_infos,
		"paper_rows": paper_rows,
		"per_fire_rows": per_fire_rows,
		"evaluated_models": evaluated_models,
		"warnings": warnings_list,
		"failures": failed_models,
		"skipped_models": skipped_models,
		"successful_models": successful_models,
		"output_files": output_files,
	}
	_write_reports(
		output_dir / "evaluation_report.md",
		output_dir / "evaluation_report.json",
		report,
		write_json=bool(args.include_json_outputs),
	)
	summary = {
		"mode": "quantitative",
		"split": args.split,
		"config": str(Path(args.config).expanduser().resolve()),
		"runs_root": str(runs_root),
		"output_dir": str(output_dir),
		"selection_metric": args.selection_metric,
		"selection_mode": args.selection_mode,
		"allow_sequence_mismatch": bool(args.allow_sequence_mismatch),
		"allow_normalization_mismatch": bool(args.allow_normalization_mismatch),
		"successful_models": successful_models,
		"skipped_models": skipped_models,
		"failed_models": failed_models,
		"output_files": output_files,
	}
	if bool(args.include_json_outputs):
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
	print(f"report: {output_dir / 'evaluation_report.md'}")
	print(f"output_dir: {output_dir}")
	return output_dir


def run_all_modes(args: argparse.Namespace) -> dict[str, Path]:
	"""Run quantitative and qualitative evaluation with the same parsed options."""

	print("Running quantitative evaluation...")
	quantitative_output_dir = run_quantitative(args)
	print("Running qualitative evaluation...")
	qualitative_output_dir = run_qualitative_mode(args)
	print(f"quantitative_output_dir: {quantitative_output_dir}")
	print(f"qualitative_output_dir: {qualitative_output_dir}")
	return {
		"quantitative": quantitative_output_dir,
		"qualitative": qualitative_output_dir,
	}


def main(argv: list[str] | None = None) -> None:
	args = build_argument_parser().parse_args(argv)
	if args.mode == "all":
		run_all_modes(args)
		return
	if args.mode == "qualitative":
		run_qualitative_mode(args)
		return
	run_quantitative(args)


if __name__ == "__main__":
	main()
