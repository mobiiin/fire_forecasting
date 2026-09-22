#!/usr/bin/env python3
"""Create publication-quality validation training-dynamics figures.

This utility reads only existing aggregate CSV files. It does not import the
training stack, load checkpoints, access datasets, or regenerate metrics.
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
from statistics import median
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml


DEFAULT_ANALYSIS_DIR = Path("artifacts/full_training/analysis")
DEFAULT_OUTPUT_DIR = DEFAULT_ANALYSIS_DIR / "paper_figures"
DEFAULT_CONFIG = Path("configs/plots/full_training_curves.yaml")
MODEL_ORDER = ("baseline", "CGP", "GA_Q2", "GPK")
EXPECTED_SEEDS = {42, 123, 2026}

METRICS: dict[str, dict[str, Any]] = {
	"dice": {
		"candidates": ("val_dice", "val_mask_dice"),
		"title": "(a) Fire-Mask Dice",
		"label": "Dice",
		"direction": "higher",
		"file_stem": "training_dice",
	},
	"energy": {
		"candidates": ("val_energy_log_mae",),
		"title": "(b) Energy Log MAE",
		"label": "Energy Log MAE",
		"direction": "lower",
		"file_stem": "training_energy_log_mae",
	},
	"surface": {
		"candidates": ("val_surface_mae", "val_surface_consumed_mae"),
		"title": "(c) Surface MAE",
		"label": "Surface MAE",
		"direction": "lower",
		"file_stem": "training_surface_mae",
	},
	"canopy": {
		"candidates": ("val_canopy_mae", "val_canopy_consumed_mae"),
		"title": "(d) Canopy MAE",
		"label": "Canopy MAE",
		"direction": "lower",
		"file_stem": "training_canopy_mae",
	},
	"active_canopy": {
		"candidates": ("val_active_canopy_mae", "val_active_canopy_consumed_mae"),
		"title": "(e) Active Canopy MAE",
		"label": "Active Canopy MAE",
		"direction": "lower",
		"file_stem": "training_active_canopy_mae",
	},
}

X_AXES = {
	"equivalent": {
		"aggregate_column": "equivalent_full_dataset_epochs",
		"raw_column": "equivalent_full_dataset_epochs",
		"label": "Equivalent Full-Dataset Passes",
	},
	"epoch": {
		"aggregate_column": "epoch",
		"raw_column": "epoch",
		"label": "Shortened Training Epoch",
	},
	"step": {
		"aggregate_column": "mean_global_step",
		"raw_column": "global_step",
		"label": "Optimizer Step",
	},
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
	for key, value in override.items():
		if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
			base[key] = _deep_merge(dict(base[key]), value)
		else:
			base[key] = value
	return base


def _read_yaml(path: Path) -> dict[str, Any]:
	if not path.is_file():
		raise FileNotFoundError(f"Plot configuration is missing: {path}")
	payload = yaml.safe_load(path.read_text(encoding="utf-8"))
	if not isinstance(payload, Mapping):
		raise ValueError(f"Plot configuration must be a mapping: {path}")
	return dict(payload)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
	if not path.is_file():
		raise FileNotFoundError(path)
	with path.open(newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		if reader.fieldnames is None:
			raise ValueError(f"CSV has no header: {path}")
		return list(reader.fieldnames), [dict(row) for row in reader]


def _float(value: Any, *, field: str) -> float:
	try:
		resolved = float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"Expected finite numeric {field}, got {value!r}.") from exc
	if not math.isfinite(resolved):
		raise ValueError(f"Expected finite numeric {field}, got {value!r}.")
	return resolved


def _optional_float(value: Any) -> float | None:
	if value in (None, "", "None", "null"):
		return None
	try:
		resolved = float(value)
	except (TypeError, ValueError):
		return None
	return resolved if math.isfinite(resolved) else None


def _sha256(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as handle:
		for chunk in iter(lambda: handle.read(1024 * 1024), b""):
			digest.update(chunk)
	return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
	temporary = path.with_suffix(path.suffix + ".tmp")
	temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
	if not rows:
		raise ValueError("Refusing to save empty plot data.")
	columns = [
		"model",
		"model_label",
		"metric",
		"source_metric",
		"x_axis",
		"x",
		"mean",
		"std",
		"n_seeds",
		"raw_mean",
		"smoothing_window",
	]
	temporary = path.with_suffix(path.suffix + ".tmp")
	with temporary.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=columns)
		writer.writeheader()
		writer.writerows(rows)
	temporary.replace(path)


def _discover_metric_columns(
	aggregate_rows: Sequence[Mapping[str, str]],
	raw_columns: Sequence[str],
) -> dict[str, str]:
	aggregated_metrics = {str(row.get("metric", "")) for row in aggregate_rows}
	raw_set = set(raw_columns)
	resolved: dict[str, str] = {}
	for key, definition in METRICS.items():
		candidates = tuple(str(value) for value in definition["candidates"])
		column = next((candidate for candidate in candidates if candidate in aggregated_metrics and candidate in raw_set), None)
		if column is None:
			column = next((candidate for candidate in candidates if candidate in raw_set), None)
		if column is None:
			raise KeyError(f"Could not resolve validation metric {key!r}; candidates={candidates}.")
		if not column.startswith("val_") or "test" in column.lower():
			raise RuntimeError(f"Unsafe non-validation metric selected: {column}")
		resolved[key] = column
	return resolved


def _validate_raw_runs(raw_rows: Sequence[Mapping[str, str]]) -> dict[str, list[int]]:
	models = {str(row.get("model_name", "")) for row in raw_rows}
	missing_models = set(MODEL_ORDER) - models
	if missing_models:
		raise RuntimeError(f"Missing required models in all_epoch_history.csv: {sorted(missing_models)}")
	seeds_by_model: dict[str, set[int]] = defaultdict(set)
	for row in raw_rows:
		model = str(row.get("model_name", ""))
		if model in MODEL_ORDER:
			seeds_by_model[model].add(int(_float(row.get("seed"), field="seed")))
	for model in MODEL_ORDER:
		if seeds_by_model[model] != EXPECTED_SEEDS:
			raise RuntimeError(
				f"Expected seeds {sorted(EXPECTED_SEEDS)} for {model}, got {sorted(seeds_by_model[model])}."
			)
	return {model: sorted(values) for model, values in seeds_by_model.items()}


def _aggregate_is_complete(rows: Sequence[Mapping[str, str]], metric_columns: Mapping[str, str]) -> bool:
	coverage = {(str(row.get("model_name")), str(row.get("metric"))) for row in rows}
	return all((model, metric_columns[key]) in coverage for model in MODEL_ORDER for key in METRICS)


def _reconstruct_aggregate(
	raw_rows: Sequence[Mapping[str, str]],
	metric_columns: Mapping[str, str],
) -> list[dict[str, Any]]:
	"""Reconstruct only required validation aggregates if the primary file is incomplete."""

	groups: dict[tuple[str, int, str], list[Mapping[str, str]]] = defaultdict(list)
	for row in raw_rows:
		model = str(row.get("model_name", ""))
		if model not in MODEL_ORDER:
			continue
		epoch = int(_float(row.get("epoch"), field="epoch"))
		for source_metric in metric_columns.values():
			if _optional_float(row.get(source_metric)) is not None:
				groups[(model, epoch, source_metric)].append(row)

	result: list[dict[str, Any]] = []
	for (model, epoch, source_metric), items in sorted(groups.items()):
		values = np.asarray([_float(item[source_metric], field=source_metric) for item in items], dtype=np.float64)
		equivalent = np.asarray(
			[_float(item["equivalent_full_dataset_epochs"], field="equivalent_full_dataset_epochs") for item in items],
			dtype=np.float64,
		)
		steps = np.asarray([_float(item["global_step"], field="global_step") for item in items], dtype=np.float64)
		result.append(
			{
				"model_name": model,
				"epoch": epoch,
				"equivalent_full_dataset_epochs": float(equivalent.mean()),
				"equivalent_full_dataset_epochs_std": float(equivalent.std(ddof=1)) if len(equivalent) > 1 else None,
				"mean_global_step": float(steps.mean()),
				"metric": source_metric,
				"mean": float(values.mean()),
				"std": float(values.std(ddof=1)) if len(values) > 1 else None,
				"min": float(values.min()),
				"max": float(values.max()),
				"number_of_seeds": len(items),
				"n_seeds_at_epoch": len(items),
			}
		)
	return result


def _centered_rolling(values: Sequence[float], window: int) -> np.ndarray:
	array = np.asarray(values, dtype=np.float64)
	if window <= 1:
		return array.copy()
	result = np.empty_like(array)
	left = (window - 1) // 2
	right = window // 2
	for index in range(len(array)):
		start = max(0, index - left)
		stop = min(len(array), index + right + 1)
		result[index] = float(array[start:stop].mean())
	return result


def _prepare_plot_rows(
	aggregate_rows: Sequence[Mapping[str, Any]],
	metric_columns: Mapping[str, str],
	config: Mapping[str, Any],
	*,
	x_axis: str,
	min_seeds: int,
	smooth: int,
) -> list[dict[str, Any]]:
	x_column = str(X_AXES[x_axis]["aggregate_column"])
	reverse_metrics = {source: key for key, source in metric_columns.items()}
	grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
	for source in aggregate_rows:
		model = str(source.get("model_name", ""))
		source_metric = str(source.get("metric", ""))
		if model not in MODEL_ORDER or source_metric not in reverse_metrics:
			continue
		if "test" in source_metric.lower() or not source_metric.startswith("val_"):
			raise RuntimeError(f"Refusing to plot non-validation metric {source_metric!r}.")
		n_seeds = int(_float(source.get("n_seeds_at_epoch", source.get("number_of_seeds")), field="n_seeds"))
		if n_seeds < min_seeds:
			continue
		std = _optional_float(source.get("std"))
		if n_seeds >= 2 and std is None:
			raise RuntimeError(f"Missing/non-finite std for {model}/{source_metric} with n={n_seeds}.")
		grouped[(model, reverse_metrics[source_metric])].append(
			{
				"model": model,
				"metric": reverse_metrics[source_metric],
				"source_metric": source_metric,
				"x": _float(source.get(x_column), field=x_column),
				"raw_mean": _float(source.get("mean"), field="mean"),
				"std": std,
				"n_seeds": n_seeds,
			}
		)

	rows: list[dict[str, Any]] = []
	seen: set[tuple[str, str, float]] = set()
	for model in MODEL_ORDER:
		for metric in METRICS:
			items = sorted(grouped.get((model, metric), []), key=lambda item: item["x"])
			if not items:
				raise RuntimeError(f"No plottable rows for {model}/{metric} at min_seeds={min_seeds}.")
			x_values = np.asarray([item["x"] for item in items], dtype=np.float64)
			if len(x_values) > 1 and not np.all(np.diff(x_values) > 0):
				raise RuntimeError(f"X values are not strictly increasing for {model}/{metric}.")
			smoothed = _centered_rolling([item["raw_mean"] for item in items], smooth)
			for item, plotted_mean in zip(items, smoothed.tolist()):
				key = (model, metric, float(item["x"]))
				if key in seen:
					raise RuntimeError(f"Duplicate model/metric/x plot row: {key}")
				seen.add(key)
				rows.append(
					{
						**item,
						"model_label": str(config["models"][model]["label"]),
						"x_axis": x_axis,
						"mean": float(plotted_mean),
						"smoothing_window": int(smooth),
					}
				)
	return rows


def _prepare_seed_curves(
	raw_rows: Sequence[Mapping[str, str]],
	metric_columns: Mapping[str, str],
	*,
	x_axis: str,
	smooth: int,
) -> dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]]:
	x_column = str(X_AXES[x_axis]["raw_column"])
	grouped: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
	for row in raw_rows:
		model = str(row.get("model_name", ""))
		if model not in MODEL_ORDER:
			continue
		seed = int(_float(row.get("seed"), field="seed"))
		x_value = _optional_float(row.get(x_column))
		if x_value is None:
			continue
		for metric, source_metric in metric_columns.items():
			value = _optional_float(row.get(source_metric))
			if value is not None:
				grouped[(model, metric, seed)].append((x_value, value))

	curves: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
	for key, values in grouped.items():
		ordered = sorted(values)
		x_values = np.asarray([value[0] for value in ordered], dtype=np.float64)
		y_values = np.asarray([value[1] for value in ordered], dtype=np.float64)
		if len(set(x_values.tolist())) != len(x_values):
			raise RuntimeError(f"Duplicate per-seed x values for {key}.")
		if len(x_values) > 1 and not np.all(np.diff(x_values) > 0):
			raise RuntimeError(f"Non-monotonic per-seed x values for {key}.")
		curves[key] = (x_values, _centered_rolling(y_values, smooth))
	return curves


def _best_locations(
	best_rows: Sequence[Mapping[str, str]],
	raw_rows: Sequence[Mapping[str, str]],
	*,
	x_axis: str,
) -> dict[str, float]:
	x_column = str(X_AXES[x_axis]["raw_column"])
	lookup: dict[tuple[str, int, int], float] = {}
	for row in raw_rows:
		model = str(row.get("model_name", ""))
		if model in MODEL_ORDER:
			key = (
				model,
				int(_float(row.get("seed"), field="seed")),
				int(_float(row.get("epoch"), field="epoch")),
			)
			lookup[key] = _float(row.get(x_column), field=x_column)
	locations: dict[str, list[float]] = defaultdict(list)
	for row in best_rows:
		model = str(row.get("model_name", ""))
		if model not in MODEL_ORDER:
			continue
		key = (
			model,
			int(_float(row.get("seed"), field="seed")),
			int(_float(row.get("best_epoch"), field="best_epoch")),
		)
		if key not in lookup:
			raise RuntimeError(f"Best epoch has no corresponding history row: {key}")
		locations[model].append(lookup[key])
	return {model: float(median(values)) for model, values in locations.items()}


def _available_font_list(configured: Sequence[str], fallback: str) -> list[str]:
	available = {item.name for item in font_manager.fontManager.ttflist}
	resolved = [str(name) for name in configured if str(name) in available]
	if fallback not in resolved:
		resolved.append(fallback)
	return resolved


def _configure_matplotlib(config: Mapping[str, Any]) -> None:
	fonts = config["fonts"]
	family = str(fonts["family"])
	serif = _available_font_list(fonts.get("serif", []), "DejaVu Serif")
	sans = _available_font_list(fonts.get("sans_serif", []), "DejaVu Sans")
	plt.rcParams.update(
		{
			"font.family": family,
			"font.serif": serif,
			"font.sans-serif": sans,
			"font.size": float(fonts["base_size"]),
			"axes.titlesize": float(fonts["title_size"]),
			"axes.labelsize": float(fonts["label_size"]),
			"xtick.labelsize": float(fonts["tick_size"]),
			"ytick.labelsize": float(fonts["tick_size"]),
			"legend.fontsize": float(fonts["legend_size"]),
			"axes.linewidth": 0.65,
			"axes.axisbelow": True,
			"axes.facecolor": "white",
			"figure.facecolor": "white",
			"savefig.facecolor": "white",
			"pdf.fonttype": 42,
			"ps.fonttype": 42,
			"svg.fonttype": "none",
			"axes.unicode_minus": True,
		}
	)


def _apply_theme(config: dict[str, Any], theme: str) -> None:
	if theme == "grayscale":
		colors = {"baseline": "#111111", "CGP": "#4D4D4D", "GA_Q2": "#808080", "GPK": "#B3B3B3"}
		for model, color in colors.items():
			config["models"][model]["color"] = color
	elif theme == "presentation":
		config["figure"].update({"width": 12.0, "height": 7.0, "dpi": 300})
		config["fonts"].update(
			{"base_size": 14, "title_size": 15, "label_size": 14, "tick_size": 12, "legend_size": 13}
		)
		config["lines"].update({"linewidth": 2.6, "seed_linewidth": 1.0, "std_alpha": 0.18})
	elif theme != "paper":
		raise ValueError(f"Unsupported theme: {theme}")


def _resolved_config(args: argparse.Namespace) -> dict[str, Any]:
	default_path = DEFAULT_CONFIG.expanduser().resolve()
	requested_path = args.config.expanduser().resolve()
	config = _read_yaml(default_path)
	if requested_path != default_path:
		config = _deep_merge(config, _read_yaml(requested_path))
	theme = args.theme or str(config.get("theme", "paper"))
	_apply_theme(config, theme)
	config["theme"] = theme
	if args.layout is not None:
		config.setdefault("layout", {})["type"] = args.layout
	if args.x_axis is not None:
		config.setdefault("x_axis", {})["type"] = args.x_axis
	if args.uncertainty is not None:
		config["uncertainty"] = args.uncertainty
	if args.smooth is not None:
		config.setdefault("smoothing", {})["window"] = args.smooth
	if args.min_seeds is not None:
		config["min_seeds"] = args.min_seeds
	if args.font_family is not None:
		config["fonts"]["family"] = args.font_family
	if args.font_size is not None:
		base = float(args.font_size)
		config["fonts"].update(
			{
				"base_size": base,
				"title_size": base + 0.5,
				"label_size": base,
				"tick_size": max(base - 1.0, 5.0),
				"legend_size": max(base - 0.5, 5.0),
			}
		)
	for argument, key in (
		(args.color_baseline, "baseline"),
		(args.color_cgp, "CGP"),
		(args.color_ga_q2, "GA_Q2"),
		(args.color_gpk, "GPK"),
	):
		if argument is not None:
			config["models"][key]["color"] = argument
	for argument, key in ((args.width, "width"), (args.height, "height"), (args.dpi, "dpi")):
		if argument is not None:
			config["figure"][key] = argument
	if args.log_y is not None:
		config["log_y"] = args.log_y
	return config


def _figure_size(config: Mapping[str, Any], layout: str) -> tuple[float, float]:
	width = float(config["figure"]["width"])
	height = float(config["figure"]["height"])
	if layout == "1x5" and width <= 8.0:
		return 11.5, max(2.8, height * 0.58)
	if layout == "5x1" and height <= 8.0:
		return min(width, 4.3), 12.0
	return width, height


def _curve_rows(rows: Sequence[Mapping[str, Any]], model: str, metric: str) -> list[Mapping[str, Any]]:
	return [row for row in rows if row["model"] == model and row["metric"] == metric]


def _set_data_driven_ylim(
	ax: Any,
	rows: Sequence[Mapping[str, Any]],
	metric: str,
	config: Mapping[str, Any],
) -> None:
	values: list[float] = []
	show_std = str(config["uncertainty"]) == "std"
	for row in rows:
		if row["metric"] != metric:
			continue
		mean = float(row["mean"])
		std = float(row["std"] or 0.0) if show_std else 0.0
		values.extend((mean - std, mean + std))
	if not values:
		return
	minimum, maximum = min(values), max(values)
	manual = config.get("ylim", {}).get(metric, [None, None])
	manual_min = None if not isinstance(manual, Sequence) or len(manual) < 1 else manual[0]
	manual_max = None if not isinstance(manual, Sequence) or len(manual) < 2 else manual[1]
	if metric in set(config.get("log_y", [])):
		positive = [value for value in values if value > 0]
		if not positive:
			raise RuntimeError(f"Cannot use log scale for non-positive {metric} data.")
		lower = float(manual_min) if manual_min is not None else min(positive) / 1.08
		upper = float(manual_max) if manual_max is not None else maximum * 1.08
	else:
		span = maximum - minimum
		padding = 0.06 * span if span > 0 else max(abs(maximum) * 0.06, 1.0e-6)
		lower = float(manual_min) if manual_min is not None else minimum - padding
		upper = float(manual_max) if manual_max is not None else maximum + padding
	if not lower < upper:
		raise ValueError(f"Invalid y limits for {metric}: {(lower, upper)}")
	ax.set_ylim(lower, upper)


def _legend_handles(config: Mapping[str, Any]) -> list[Line2D]:
	return [
		Line2D(
			[0],
			[0],
			color=config["models"][model]["color"],
			linestyle=config["models"][model]["linestyle"],
			linewidth=float(config["lines"]["linewidth"]),
			label=config["models"][model]["label"],
		)
		for model in MODEL_ORDER
	]


def _plot_panel(
	ax: Any,
	metric: str,
	plot_rows: Sequence[Mapping[str, Any]],
	seed_curves: Mapping[tuple[str, str, int], tuple[np.ndarray, np.ndarray]],
	best_locations: Mapping[str, float],
	config: Mapping[str, Any],
	*,
	x_axis: str,
	show_seeds: bool,
	show_best: bool,
	annotate_final: bool,
	show_x_label: bool,
) -> None:
	line_config = config["lines"]
	for model_index, model in enumerate(MODEL_ORDER):
		style = config["models"][model]
		if show_seeds:
			for seed in sorted(EXPECTED_SEEDS):
				curve = seed_curves.get((model, metric, seed))
				if curve is not None:
					ax.plot(
						curve[0],
						curve[1],
						color=style["color"],
						linestyle=style["linestyle"],
						linewidth=float(line_config.get("seed_linewidth", 0.7)),
						alpha=float(line_config.get("seed_alpha", 0.22)),
						zorder=1,
					)
		rows = _curve_rows(plot_rows, model, metric)
		x_values = np.asarray([float(row["x"]) for row in rows], dtype=np.float64)
		means = np.asarray([float(row["mean"]) for row in rows], dtype=np.float64)
		stds = np.asarray([float(row["std"] or 0.0) for row in rows], dtype=np.float64)
		if str(config["uncertainty"]) == "std":
			ax.fill_between(
				x_values,
				means - stds,
				means + stds,
				color=style["color"],
				alpha=float(line_config["std_alpha"]),
				linewidth=0,
				zorder=2,
			)
		ax.plot(
			x_values,
			means,
			color=style["color"],
			linestyle=style["linestyle"],
			linewidth=float(line_config["linewidth"]),
			marker=style.get("marker"),
			zorder=3,
		)
		if show_best and model in best_locations:
			best_x = float(best_locations[model])
			if x_values[0] <= best_x <= x_values[-1]:
				best_y = float(np.interp(best_x, x_values, means))
				ax.plot(
					[best_x],
					[best_y],
					marker="o",
					markersize=3.5,
					markerfacecolor="white",
					markeredgecolor=style["color"],
					markeredgewidth=0.9,
					zorder=4,
				)
		if annotate_final:
			offsets = ((3, 4), (3, -8), (3, 7), (3, -11))
			ax.annotate(
				f"{means[-1]:.3g}",
				xy=(x_values[-1], means[-1]),
				xytext=offsets[model_index],
				textcoords="offset points",
				fontsize=max(float(config["fonts"]["tick_size"]) - 1.0, 5.0),
				color=style["color"],
				clip_on=True,
			)

	ax.set_title(str(METRICS[metric]["title"]), pad=4.0)
	ax.set_ylabel(str(METRICS[metric]["label"]), labelpad=3.0)
	if show_x_label:
		ax.set_xlabel(str(X_AXES[x_axis]["label"]), labelpad=3.0)
	if metric in set(config.get("log_y", [])):
		ax.set_yscale("log")
	grid = config.get("grid", {})
	if bool(grid.get("enabled", True)):
		ax.grid(
			True,
			axis="y",
			color="#808080",
			alpha=float(grid.get("alpha", 0.2)),
			linewidth=float(grid.get("linewidth", 0.5)),
		)
	ax.grid(False, axis="x")
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_color("#555555")
	ax.spines["bottom"].set_color("#555555")
	ax.tick_params(width=0.6, length=3.0, color="#555555")
	if x_axis == "step":
		ax.ticklabel_format(axis="x", style="sci", scilimits=(4, 4), useMathText=True)
	if config.get("x_min") is not None or config.get("x_max") is not None:
		left, right = ax.get_xlim()
		ax.set_xlim(
			float(config["x_min"]) if config.get("x_min") is not None else left,
			float(config["x_max"]) if config.get("x_max") is not None else right,
		)
	_set_data_driven_ylim(ax, plot_rows, metric, config)


def _save_figure(fig: Any, output_dir: Path, stem: str, dpi: int, *, svg: bool) -> None:
	fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
	if svg:
		fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
	fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")


def _render_combined(
	plot_rows: Sequence[Mapping[str, Any]],
	seed_curves: Mapping[tuple[str, str, int], tuple[np.ndarray, np.ndarray]],
	best_locations: Mapping[str, float],
	config: Mapping[str, Any],
	output_dir: Path,
	*,
	stem: str,
	show_seeds: bool,
) -> tuple[float, float]:
	layout = str(config["layout"]["type"])
	if layout not in {"2x3", "1x5", "5x1"}:
		raise ValueError(f"Unsupported layout: {layout}")
	figsize = _figure_size(config, layout)
	if layout == "2x3":
		fig, axes_array = plt.subplots(2, 3, figsize=figsize)
		axes = list(axes_array.flat)
		data_axes = axes[:5]
		legend_axis = axes[5]
		legend_axis.axis("off")
		legend_axis.legend(
			handles=_legend_handles(config),
			loc=str(config["legend"].get("location", "center")),
			ncol=int(config["legend"].get("columns", 1)),
			frameon=bool(config["legend"].get("frame", False)),
			handlelength=2.8,
		)
		x_label_indices = {3, 4}
		fig.subplots_adjust(left=0.09, right=0.985, bottom=0.105, top=0.965, wspace=0.36, hspace=0.38)
	elif layout == "1x5":
		fig, axes_array = plt.subplots(1, 5, figsize=figsize)
		data_axes = list(np.asarray(axes_array).flat)
		x_label_indices = set(range(5))
		fig.legend(
			handles=_legend_handles(config),
			loc="lower center",
			bbox_to_anchor=(0.5, 0.01),
			ncol=4,
			frameon=bool(config["legend"].get("frame", False)),
			handlelength=2.8,
		)
		fig.subplots_adjust(left=0.055, right=0.995, bottom=0.27, top=0.91, wspace=0.42)
	else:
		fig, axes_array = plt.subplots(5, 1, figsize=figsize)
		data_axes = list(np.asarray(axes_array).flat)
		x_label_indices = {4}
		fig.legend(
			handles=_legend_handles(config),
			loc="upper center",
			bbox_to_anchor=(0.5, 0.995),
			ncol=2,
			frameon=bool(config["legend"].get("frame", False)),
			handlelength=2.8,
		)
		fig.subplots_adjust(left=0.19, right=0.98, bottom=0.07, top=0.91, hspace=0.48)

	for index, metric in enumerate(METRICS):
		_plot_panel(
			data_axes[index],
			metric,
			plot_rows,
			seed_curves,
			best_locations,
			config,
			x_axis=str(config["x_axis"]["type"]),
			show_seeds=show_seeds,
			show_best=bool(config.get("show_best", False)),
			annotate_final=bool(config.get("annotate_final", False)),
			show_x_label=index in x_label_indices,
		)
	fig.align_ylabels(data_axes)
	_save_figure(fig, output_dir, stem, int(config["figure"]["dpi"]), svg=stem == "training_dynamics")
	plt.close(fig)
	return figsize


def _render_individual_panels(
	plot_rows: Sequence[Mapping[str, Any]],
	seed_curves: Mapping[tuple[str, str, int], tuple[np.ndarray, np.ndarray]],
	best_locations: Mapping[str, float],
	config: Mapping[str, Any],
	output_dir: Path,
	*,
	show_seeds: bool,
) -> None:
	for metric, definition in METRICS.items():
		fig, ax = plt.subplots(figsize=(3.5, 2.75))
		_plot_panel(
			ax,
			metric,
			plot_rows,
			seed_curves,
			best_locations,
			config,
			x_axis=str(config["x_axis"]["type"]),
			show_seeds=show_seeds,
			show_best=bool(config.get("show_best", False)),
			annotate_final=bool(config.get("annotate_final", False)),
			show_x_label=True,
		)
		fig.legend(
			handles=_legend_handles(config),
			loc="lower center",
			bbox_to_anchor=(0.5, 0.005),
			ncol=2,
			frameon=False,
			handlelength=2.5,
		)
		fig.subplots_adjust(left=0.18, right=0.98, bottom=0.31, top=0.88)
		_save_figure(fig, output_dir, str(definition["file_stem"]), int(config["figure"]["dpi"]), svg=False)
		plt.close(fig)


def _parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
	parser.add_argument("--output-dir", type=Path, default=None)
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
	parser.add_argument("--theme", choices=("paper", "grayscale", "presentation"), default=None)
	parser.add_argument("--x-axis", choices=tuple(X_AXES), default=None)
	parser.add_argument("--uncertainty", choices=("std", "none"), default=None)
	parser.add_argument("--smooth", type=int, default=None, help="Centered rolling-mean window; 0 disables smoothing.")
	parser.add_argument("--min-seeds", type=int, choices=(1, 2, 3), default=None)
	parser.add_argument("--layout", choices=("2x3", "1x5", "5x1"), default=None)
	parser.add_argument("--font-family", choices=("serif", "sans-serif"), default=None)
	parser.add_argument("--font-size", type=float, default=None)
	parser.add_argument("--width", type=float, default=None)
	parser.add_argument("--height", type=float, default=None)
	parser.add_argument("--dpi", type=int, default=None)
	parser.add_argument("--x-min", type=float, default=None)
	parser.add_argument("--x-max", type=float, default=None)
	parser.add_argument("--show-best", action="store_true")
	parser.add_argument("--show-seeds", action="store_true")
	parser.add_argument("--supplementary-with-seeds", action="store_true")
	parser.add_argument("--annotate-final", action="store_true")
	parser.add_argument("--log-y", action="append", choices=tuple(METRICS), default=None)
	parser.add_argument("--color-baseline", default=None)
	parser.add_argument("--color-cgp", default=None)
	parser.add_argument("--color-ga-q2", default=None)
	parser.add_argument("--color-gpk", default=None)
	return parser


def main() -> None:
	args = _parser().parse_args()
	if args.smooth is not None and args.smooth < 0:
		raise ValueError("--smooth must be >= 0.")
	if args.x_min is not None and args.x_max is not None and args.x_min >= args.x_max:
		raise ValueError("--x-min must be less than --x-max.")

	config = _resolved_config(args)
	config["show_best"] = bool(args.show_best)
	config["annotate_final"] = bool(args.annotate_final)
	config["x_min"] = args.x_min
	config["x_max"] = args.x_max
	x_axis = str(config["x_axis"]["type"])
	if x_axis not in X_AXES:
		raise ValueError(f"Unsupported x-axis type: {x_axis}")
	uncertainty = str(config.get("uncertainty", "std"))
	if uncertainty not in {"std", "none"}:
		raise ValueError(f"Unsupported uncertainty type: {uncertainty}")
	config["uncertainty"] = uncertainty
	smooth = int(config.get("smoothing", {}).get("window", 0))
	if smooth < 0:
		raise ValueError("smoothing.window must be >= 0.")
	min_seeds = int(config.get("min_seeds", 2))
	if min_seeds not in {1, 2, 3}:
		raise ValueError("min_seeds must be 1, 2, or 3.")

	analysis_dir = args.analysis_dir.expanduser().resolve()
	output_dir = (
		args.output_dir.expanduser().resolve()
		if args.output_dir is not None
		else analysis_dir / "paper_figures"
	)
	output_dir.mkdir(parents=True, exist_ok=True)
	aggregate_path = analysis_dir / "learning_curves_mean_std.csv"
	raw_path = analysis_dir / "all_epoch_history.csv"
	best_path = analysis_dir / "best_epochs.csv"
	aggregate_columns, aggregate_rows = _read_csv(aggregate_path)
	raw_columns, raw_rows = _read_csv(raw_path)
	best_columns, best_rows = _read_csv(best_path)
	_ = aggregate_columns, best_columns
	metric_columns = _discover_metric_columns(aggregate_rows, raw_columns)
	seeds_by_model = _validate_raw_runs(raw_rows)
	primary_source = aggregate_path
	reconstructed = False
	if not _aggregate_is_complete(aggregate_rows, metric_columns):
		aggregate_rows = _reconstruct_aggregate(raw_rows, metric_columns)
		primary_source = raw_path
		reconstructed = True

	plot_rows = _prepare_plot_rows(
		aggregate_rows,
		metric_columns,
		config,
		x_axis=x_axis,
		min_seeds=min_seeds,
		smooth=smooth,
	)
	seed_curves = _prepare_seed_curves(raw_rows, metric_columns, x_axis=x_axis, smooth=smooth)
	best_locations = _best_locations(best_rows, raw_rows, x_axis=x_axis) if args.show_best else {}
	_configure_matplotlib(config)
	main_size = _render_combined(
		plot_rows,
		seed_curves,
		best_locations,
		config,
		output_dir,
		stem="training_dynamics",
		show_seeds=bool(args.show_seeds),
	)
	_render_individual_panels(
		plot_rows,
		seed_curves,
		best_locations,
		config,
		output_dir,
		show_seeds=bool(args.show_seeds),
	)
	if args.supplementary_with_seeds:
		_render_combined(
			plot_rows,
			seed_curves,
			best_locations,
			config,
			output_dir,
			stem="training_dynamics_with_seeds",
			show_seeds=True,
		)

	plot_data_path = output_dir / "training_dynamics_plot_data.csv"
	_atomic_csv(plot_data_path, plot_rows)
	metadata_path = output_dir / "training_dynamics_plot_metadata.json"
	default_config_path = DEFAULT_CONFIG.expanduser().resolve()
	requested_config_path = args.config.expanduser().resolve()
	source_paths = [aggregate_path, raw_path, best_path, default_config_path]
	if requested_config_path != default_config_path:
		source_paths.append(requested_config_path)
	validation_scopes = sorted(
		{str(row.get("validation_scope")) for row in raw_rows if row.get("validation_scope") not in (None, "")}
	)
	metadata = {
		"schema_version": 1,
		"generated_at": datetime.now(timezone.utc).isoformat(),
		"analysis_directory": str(analysis_dir),
		"output_directory": str(output_dir),
		"primary_source_file": str(primary_source),
		"aggregate_reconstructed_from_raw": reconstructed,
		"source_files": [
			{"path": str(path), "sha256": _sha256(path)} for path in source_paths
		],
		"selected_models": list(MODEL_ORDER),
		"model_labels": {model: config["models"][model]["label"] for model in MODEL_ORDER},
		"seeds_by_model": seeds_by_model,
		"metrics": {
			key: {
				"source_column": metric_columns[key],
				"title": definition["title"],
				"direction": definition["direction"],
			}
			for key, definition in METRICS.items()
		},
		"x_axis_type": x_axis,
		"x_axis_column": X_AXES[x_axis]["aggregate_column"],
		"x_axis_label": X_AXES[x_axis]["label"],
		"smoothing_window": smooth,
		"smoothing_definition": (
			"disabled"
			if smooth <= 1
			else "centered rolling mean applied to mean/seed lines; raw per-coordinate std is not smoothed"
		),
		"uncertainty": uncertainty,
		"uncertainty_definition": "mean plus/minus one sample standard deviation across available seeds",
		"min_seeds": min_seeds,
		"layout": config["layout"]["type"],
		"figure_size_inches": list(main_size),
		"dpi": int(config["figure"]["dpi"]),
		"show_seed_curves": bool(args.show_seeds),
		"show_best_epoch_markers": bool(args.show_best),
		"annotate_final": bool(args.annotate_final),
		"log_y": list(config.get("log_y", [])),
		"plot_data_rows": len(plot_rows),
		"matplotlib_version": matplotlib.__version__,
		"numpy_version": np.__version__,
		"plotting_config": config,
		"scientific_scope": "validation training dynamics only; no test-set metrics",
		"validation_scopes_in_source": validation_scopes,
	}
	_atomic_json(metadata_path, metadata)

	print("SOURCE ANALYSIS DIRECTORY:")
	print(f"  {analysis_dir}")
	print("PRIMARY SOURCE FILE:")
	print(f"  {primary_source}")
	print("DETECTED MODELS:")
	print("  " + ", ".join(MODEL_ORDER))
	print("DETECTED METRICS:")
	print("  " + ", ".join(f"{key}={value}" for key, value in metric_columns.items()))
	print("DETECTED X AXES:")
	print("  equivalent_full_dataset_epochs, epoch, global_step")
	print("DEFAULT X AXIS:" if args.x_axis is None else "SELECTED X AXIS:")
	print(f"  {X_AXES[x_axis]['label']}")
	print("OUTPUT DIRECTORY:")
	print(f"  {output_dir}")
	print("GENERATED:")
	generated_names = [
		"training_dynamics.pdf",
		"training_dynamics.svg",
		"training_dynamics.png",
		*[f"{definition['file_stem']}.pdf" for definition in METRICS.values()],
		"training_dynamics_plot_data.csv",
		"training_dynamics_plot_metadata.json",
	]
	if args.supplementary_with_seeds:
		generated_names.extend(["training_dynamics_with_seeds.pdf", "training_dynamics_with_seeds.png"])
	for name in generated_names:
		print(f"  {name}")
	print("REGENERATE:")
	print("  python scripts/plot_full_training_curves.py")


if __name__ == "__main__":
	main()

