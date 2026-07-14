"""Discover and select trained model runs for paper evaluation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable
import warnings


COMPLETED_STATUSES = {"completed", "complete", "succeeded", "success", "done"}
MINIMIZE_TOKENS = ("loss", "mae", "rmse", "error")
MAXIMIZE_TOKENS = ("dice", "iou", "skill", "accuracy", "precision", "recall")


@dataclass(frozen=True)
class TrainingRun:
	"""Metadata for one artifacts/runs/<architecture>/<run_name> directory."""

	architecture: str
	run_name: str
	run_dir: str
	status: str
	checkpoint_path: str
	config_path: str | None
	resolved_config_path: str | None
	best_metric_name: str | None
	best_metric_value: float | None
	best_epoch: int | None
	final_val_loss: float | None
	metadata_partial: bool
	summary_path: str | None
	metrics: dict[str, float]

	def to_dict(self) -> dict[str, Any]:
		"""Return JSON/CSV-friendly run metadata."""

		return asdict(self)


def _safe_float(value: Any) -> float | None:
	if value in (None, ""):
		return None
	try:
		number = float(value)
	except (TypeError, ValueError):
		return None
	return number if math.isfinite(number) else None


def _safe_int(value: Any) -> int | None:
	if value in (None, ""):
		return None
	try:
		return int(value)
	except (TypeError, ValueError):
		return None


def _load_json(path: Path) -> dict[str, Any]:
	if not path.exists():
		return {}
	try:
		with path.open("r", encoding="utf-8") as handle:
			payload = json.load(handle)
	except (OSError, json.JSONDecodeError):
		return {}
	return payload if isinstance(payload, dict) else {}


def _read_last_training_metrics(path: Path) -> dict[str, float]:
	if not path.exists():
		return {}
	try:
		with path.open("r", newline="", encoding="utf-8") as handle:
			rows = list(csv.DictReader(handle))
	except OSError:
		return {}
	if not rows:
		return {}
	metrics: dict[str, float] = {}
	for key, value in rows[-1].items():
		number = _safe_float(value)
		if number is not None:
			metrics[str(key)] = number
	return metrics


def _path_from_summary(value: Any, run_dir: Path) -> Path | None:
	if value in (None, ""):
		return None
	path = Path(str(value)).expanduser()
	if not path.is_absolute():
		path = run_dir / path
	return path.resolve()


def _resolve_checkpoint_path(summary: dict[str, Any], run_dir: Path, checkpoint_name: str) -> Path:
	summary_path = _path_from_summary(summary.get("best_checkpoint_path"), run_dir)
	if summary_path is not None and summary_path.exists():
		return summary_path
	return (run_dir / "checkpoints" / checkpoint_name).resolve()


def _resolve_config_path(summary: dict[str, Any], run_dir: Path, key: str, fallback_name: str) -> str | None:
	summary_path = _path_from_summary(summary.get(key), run_dir)
	if summary_path is not None and summary_path.exists():
		return str(summary_path)
	fallback = run_dir / "configs" / fallback_name
	return str(fallback.resolve()) if fallback.exists() else None


def _candidate_run_dirs(runs_root: Path, architecture: str | None) -> Iterable[Path]:
	if architecture:
		search_root = runs_root / architecture
		if not search_root.exists():
			return []
		return sorted(path for path in search_root.glob("*") if path.is_dir())
	return sorted(path for path in runs_root.glob("*/*") if path.is_dir())


def discover_runs(runs_root: str | Path, architecture: str | None = None, checkpoint_name: str = "best_model.pt") -> list[TrainingRun]:
	"""Return valid trained runs under ``runs_root``.

	A run is considered evaluable when it has the requested checkpoint. Metadata
	from ``metadata/run_summary.json`` is preferred, but older/partial runs are
	still returned when the checkpoint exists.
	"""

	root = Path(runs_root).expanduser().resolve()
	runs: list[TrainingRun] = []
	for run_dir in _candidate_run_dirs(root, architecture):
		summary_path = run_dir / "metadata" / "run_summary.json"
		summary = _load_json(summary_path)
		checkpoint_path = _resolve_checkpoint_path(summary, run_dir, checkpoint_name)
		if not checkpoint_path.exists():
			continue
		metrics = _read_last_training_metrics(run_dir / "logs" / "training_log.csv")
		best_metric_value = _safe_float(summary.get("best_metric_value"))
		final_val_loss = _safe_float(summary.get("final_val_loss"))
		if best_metric_value is not None:
			metrics["best_metric_value"] = best_metric_value
			metrics["best_metric"] = best_metric_value
		if final_val_loss is not None:
			metrics["final_val_loss"] = final_val_loss
		best_metric_name = summary.get("best_metric_name")
		if best_metric_name not in (None, "") and best_metric_value is not None:
			metrics[str(best_metric_name)] = best_metric_value
		run = TrainingRun(
			architecture=str(summary.get("architecture") or run_dir.parent.name),
			run_name=str(summary.get("run_name") or run_dir.name),
			run_dir=str(run_dir.resolve()),
			status=str(summary.get("status") or "unknown").lower(),
			checkpoint_path=str(checkpoint_path),
			config_path=_resolve_config_path(summary, run_dir, "config_path", "original_config.yaml"),
			resolved_config_path=_resolve_config_path(summary, run_dir, "resolved_config_path", "resolved_config.yaml"),
			best_metric_name=str(best_metric_name) if best_metric_name not in (None, "") else None,
			best_metric_value=best_metric_value,
			best_epoch=_safe_int(summary.get("best_epoch")),
			final_val_loss=final_val_loss,
			metadata_partial=not bool(summary),
			summary_path=str(summary_path.resolve()) if summary_path.exists() else None,
			metrics=metrics,
		)
		runs.append(run)
	return runs


def selection_mode_for_metric(selection_metric: str, selection_mode: str = "auto") -> str:
	"""Resolve min/max selection direction for a metric name."""

	mode = str(selection_mode).lower()
	if mode in {"min", "max"}:
		return mode
	if mode != "auto":
		raise ValueError(f"selection_mode must be auto, min, or max. Got {selection_mode!r}.")
	metric = str(selection_metric).lower()
	if any(token in metric for token in MAXIMIZE_TOKENS):
		return "max"
	if any(token in metric for token in MINIMIZE_TOKENS):
		return "min"
	return "min"


def _normalized_metric_lookup(metrics: dict[str, float]) -> dict[str, float]:
	return {str(key).lower(): value for key, value in metrics.items()}


def _metric_candidates(run: TrainingRun, selection_metric: str) -> list[str]:
	metric = str(selection_metric).lower()
	if metric == "best_metric":
		candidates = ["best_metric_value", "best_metric"]
		if run.best_metric_name:
			candidates.append(str(run.best_metric_name))
		candidates.append("final_val_loss")
		return candidates
	aliases = {
		"val_loss": ["final_val_loss", "val_loss"],
		"val_multitask_loss": ["val_multitask_loss", "final_val_loss", "val_loss"],
		"val_energy_active_mae": ["val_energy_active_mae", "val_energy_mw_active_mae", "val_energy_MW_active_mae"],
		"val_dice": ["val_dice", "val_mask_dice", "mask_dice"],
		"val_iou": ["val_iou", "val_mask_iou", "mask_iou"],
	}
	return aliases.get(metric, [metric])


def metric_value_for_run(run: TrainingRun, selection_metric: str) -> tuple[float | None, str | None]:
	"""Return the metric value/name used to rank one run."""

	lookup = _normalized_metric_lookup(run.metrics)
	for candidate in _metric_candidates(run, selection_metric):
		value = lookup.get(str(candidate).lower())
		if value is not None and math.isfinite(float(value)):
			return float(value), str(candidate)
	return None, None


def find_best_run(
	runs: Iterable[TrainingRun],
	architecture: str,
	selection_metric: str = "best_metric",
	selection_mode: str = "auto",
) -> TrainingRun | None:
	"""Select the best evaluable run for one architecture."""

	architecture_name = str(architecture).lower()
	matching = [run for run in runs if run.architecture.lower() == architecture_name]
	if not matching:
		return None
	completed = [run for run in matching if run.status.lower() in COMPLETED_STATUSES]
	candidates = completed if completed else matching
	if not completed:
		warnings.warn(
			f"No completed runs found for {architecture_name}; selecting from incomplete/unknown runs with checkpoints.",
			RuntimeWarning,
			stacklevel=2,
		)
	mode = selection_mode_for_metric(selection_metric, selection_mode)
	scored: list[tuple[float, TrainingRun]] = []
	for run in candidates:
		value, _metric_name = metric_value_for_run(run, selection_metric)
		if value is not None:
			scored.append((value, run))
	if scored:
		reverse = mode == "max"
		return sorted(scored, key=lambda item: item[0], reverse=reverse)[0][1]
	warnings.warn(
		f"No finite {selection_metric!r} values found for {architecture_name}; falling back to newest modified run directory.",
		RuntimeWarning,
		stacklevel=2,
	)
	return max(candidates, key=lambda run: Path(run.run_dir).stat().st_mtime)
