"""Training-run plotting helpers."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Mapping

if "MPLCONFIGDIR" not in os.environ:
	_mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "fire_forecasting_mplconfig"
	_mpl_config_dir.mkdir(parents=True, exist_ok=True)
	os.environ["MPLCONFIGDIR"] = str(_mpl_config_dir)
if "XDG_CACHE_HOME" not in os.environ:
	_xdg_cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "fire_forecasting_xdg_cache"
	_xdg_cache_dir.mkdir(parents=True, exist_ok=True)
	os.environ["XDG_CACHE_HOME"] = str(_xdg_cache_dir)

try:
	import matplotlib
	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional in minimal environments
	matplotlib = None
	plt = None


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
	if not path.exists():
		return []
	with path.open("r", newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		return [dict(row) for row in reader]


def _finite_float(value: Any) -> float | None:
	if value in (None, "", "nan", "NaN", "None"):
		return None
	try:
		result = float(value)
	except (TypeError, ValueError):
		return None
	if result != result:
		return None
	return result


def _epoch_values(rows: list[Mapping[str, Any]], key: str) -> tuple[list[int], list[float]]:
	epochs: list[int] = []
	values: list[float] = []
	for index, row in enumerate(rows, start=1):
		value = _finite_float(row.get(key))
		if value is None:
			continue
		try:
			epoch = int(float(row.get("epoch", index)))
		except (TypeError, ValueError):
			epoch = index
		epochs.append(epoch)
		values.append(value)
	return epochs, values


def _title(architecture: str | None, run_name: str | None, label: str) -> str:
	arch_title = (architecture or "Training").replace("_", "-")
	if not run_name:
		return f"{arch_title} {label}"
	return f"{arch_title} {label}\n{run_name}"


def save_loss_curves(
	rows: list[Mapping[str, Any]],
	output_path: str | Path,
	architecture: str | None = None,
	run_name: str | None = None,
	test_results: Mapping[str, Any] | None = None,
) -> list[Path]:
	"""Save train/validation loss curves as PNG and PDF."""

	if plt is None or not rows:
		return []

	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	fig, axis = plt.subplots(1, 1, figsize=(9, 5), dpi=150, constrained_layout=True)
	for key, color, label in (
		("train_loss", "tab:blue", "Train loss"),
		("val_loss", "tab:orange", "Validation loss"),
	):
		epochs, values = _epoch_values(rows, key)
		if values:
			axis.plot(epochs, values, color=color, linewidth=2.0, marker="o", markersize=3, label=label)

	test_results = test_results or {}
	test_loss = _finite_float(test_results.get("test_loss"))
	if test_loss is not None:
		axis.axhline(test_loss, color="tab:green", linestyle="--", linewidth=1.8, label="Test loss")

	best_epochs, val_values = _epoch_values(rows, "val_loss")
	if val_values:
		best_index = min(range(len(val_values)), key=lambda index: val_values[index])
		axis.axvline(best_epochs[best_index], color="0.25", linestyle=":", linewidth=1.2, label=f"Best val epoch {best_epochs[best_index]}")

	axis.set_title(_title(architecture, run_name, "Training Curves"))
	axis.set_xlabel("Epoch")
	axis.set_ylabel("Loss")
	axis.grid(True, alpha=0.3)
	axis.legend()
	fig.savefig(output, bbox_inches="tight")
	pdf_path = output.with_suffix(".pdf")
	fig.savefig(pdf_path, bbox_inches="tight")
	plt.close(fig)
	return [output, pdf_path]


def _available_metric_names(rows: list[Mapping[str, Any]], test_results: Mapping[str, Any]) -> list[str]:
	metrics: set[str] = set()
	for row in rows:
		for key in row:
			if key.startswith(("train_", "val_")) and not key.endswith("loss") and key not in {
				"train_samples",
				"train_batches",
				"val_samples",
				"val_batches",
				"train_samples_per_second",
				"val_samples_per_second",
			}:
				metrics.add(key.split("_", 1)[1])
	for key in test_results:
		if key.startswith("test_") and not key.endswith("loss"):
			metrics.add(key.split("_", 1)[1])
	preferred = [
		"mask_dice",
		"mask_iou",
		"energy_log_mae",
		"surface_consumed_mae",
		"canopy_consumed_mae",
		"accuracy",
		"dice",
		"iou",
		"mae",
	]
	ordered = [name for name in preferred if name in metrics]
	ordered.extend(name for name in sorted(metrics) if name not in ordered)
	return ordered[:6]


def save_metric_curves(
	rows: list[Mapping[str, Any]],
	output_path: str | Path,
	architecture: str | None = None,
	run_name: str | None = None,
	test_results: Mapping[str, Any] | None = None,
) -> list[Path]:
	"""Save compact metric curves for available train/validation/test metrics."""

	if plt is None or not rows:
		return []

	test_results = test_results or {}
	metrics = _available_metric_names(rows, test_results)
	if not metrics:
		return []

	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	ncols = 2
	nrows = (len(metrics) + 1) // 2
	fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4, 3.5 * nrows)), dpi=150, constrained_layout=True)
	flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
	for axis, metric_name in zip(flat_axes, metrics):
		for prefix, color, label in (("train", "tab:blue", "Train"), ("val", "tab:orange", "Validation")):
			key = f"{prefix}_{metric_name}"
			epochs, values = _epoch_values(rows, key)
			if values:
				axis.plot(epochs, values, color=color, linewidth=2.0, marker="o", markersize=3, label=label)
		test_value = _finite_float(test_results.get(f"test_{metric_name}"))
		if test_value is not None:
			axis.axhline(test_value, color="tab:green", linestyle="--", linewidth=1.5, label="Test")
		axis.set_title(metric_name.replace("_", " ").title())
		axis.set_xlabel("Epoch")
		axis.grid(True, alpha=0.3)
		axis.legend()
	for axis in flat_axes[len(metrics):]:
		axis.axis("off")
	fig.suptitle(_title(architecture, run_name, "Metric Curves"), fontsize=14)
	fig.savefig(output, bbox_inches="tight")
	plt.close(fig)
	return [output]


def save_learning_rate_curve(
	rows: list[Mapping[str, Any]],
	output_path: str | Path,
	architecture: str | None = None,
	run_name: str | None = None,
) -> list[Path]:
	"""Save the learning-rate schedule curve when present."""

	if plt is None or not rows:
		return []
	epochs, values = _epoch_values(rows, "learning_rate")
	if not values:
		return []
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	fig, axis = plt.subplots(1, 1, figsize=(8, 4.5), dpi=150, constrained_layout=True)
	axis.plot(epochs, values, color="tab:purple", linewidth=2.0, marker="o", markersize=3)
	axis.set_title(_title(architecture, run_name, "Learning Rate"))
	axis.set_xlabel("Epoch")
	axis.set_ylabel("Learning rate")
	axis.grid(True, alpha=0.3)
	fig.savefig(output, bbox_inches="tight")
	plt.close(fig)
	return [output]


def save_timing_breakdown(
	timing_log_path: str | Path,
	output_path: str | Path,
	architecture: str | None = None,
	run_name: str | None = None,
) -> list[Path]:
	"""Save a timing breakdown from the batch-level timing CSV."""

	if plt is None:
		return []
	rows = _read_csv_rows(Path(timing_log_path))
	if not rows:
		return []
	timing_keys = ["data_wait", "h2d", "norm", "forward", "backward", "metrics", "optimizer"]
	phase_values: dict[str, dict[str, list[float]]] = {}
	for row in rows:
		phase = str(row.get("phase", "unknown") or "unknown")
		phase_values.setdefault(phase, {key: [] for key in timing_keys})
		for key in timing_keys:
			value = _finite_float(row.get(key))
			if value is not None:
				phase_values[phase][key].append(value)
	if not phase_values:
		return []
	phases = sorted(phase_values)
	x_positions = list(range(len(timing_keys)))
	width = 0.8 / max(1, len(phases))
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	fig, axis = plt.subplots(1, 1, figsize=(10, 5), dpi=150, constrained_layout=True)
	for phase_index, phase in enumerate(phases):
		means = [
			sum(phase_values[phase][key]) / len(phase_values[phase][key]) if phase_values[phase][key] else 0.0
			for key in timing_keys
		]
		offsets = [position - 0.4 + width / 2.0 + phase_index * width for position in x_positions]
		axis.bar(offsets, means, width=width, label=phase)
	axis.set_xticks(x_positions)
	axis.set_xticklabels(timing_keys, rotation=30, ha="right")
	axis.set_ylabel("Average seconds per batch")
	axis.set_title(_title(architecture, run_name, "Timing Breakdown"))
	axis.grid(True, axis="y", alpha=0.3)
	axis.legend()
	fig.savefig(output, bbox_inches="tight")
	plt.close(fig)
	return [output]


def save_training_run_figures(
	run_dir: str | Path,
	architecture: str | None = None,
	run_name: str | None = None,
	test_results: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
	"""Regenerate all standard figures for a run directory."""

	run_path = Path(run_dir)
	rows = _read_csv_rows(run_path / "logs" / "training_log.csv")
	figures: dict[str, list[str]] = {
		"loss_curves": [],
		"metric_curves": [],
		"learning_rate_curve": [],
		"timing_breakdown": [],
	}
	if rows:
		figures["loss_curves"] = [
			str(path)
			for path in save_loss_curves(rows, run_path / "figures" / "loss_curves.png", architecture, run_name, test_results)
		]
		figures["metric_curves"] = [
			str(path)
			for path in save_metric_curves(rows, run_path / "figures" / "metric_curves.png", architecture, run_name, test_results)
		]
		figures["learning_rate_curve"] = [
			str(path)
			for path in save_learning_rate_curve(rows, run_path / "figures" / "learning_rate_curve.png", architecture, run_name)
		]
	figures["timing_breakdown"] = [
		str(path)
		for path in save_timing_breakdown(run_path / "logs" / "timing_log.csv", run_path / "figures" / "timing_breakdown.png", architecture, run_name)
	]
	return figures
