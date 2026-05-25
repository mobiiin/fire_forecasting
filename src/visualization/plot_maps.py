"""Static map plotting helpers for wildfire forecast visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def _as_2d_array(array) -> np.ndarray:
	"""Convert an input array or tensor-like object to a 2D NumPy array."""

	if hasattr(array, "detach"):
		array = array.detach().cpu().numpy()
	else:
		array = np.asarray(array)

	if array.ndim == 3 and array.shape[0] == 1:
		array = array[0]
	if array.ndim != 2:
		raise ValueError(f"Expected a 2D array, got shape {array.shape}.")
	return np.asarray(array, dtype=np.float32)


def inverse_normalize_channel_map(
	array,
	channel_index: int,
	normalization_stats: Mapping[str, np.ndarray] | None,
) -> np.ndarray:
	"""Undo z-score normalization for a single channel map when stats are available."""

	array_2d = _as_2d_array(array)
	if normalization_stats is None:
		return array_2d

	mean = np.asarray(normalization_stats["mean"])[channel_index]
	std = np.asarray(normalization_stats["std"])[channel_index]
	return array_2d * std + mean


def _finite_min_max(*arrays: np.ndarray) -> tuple[float, float]:
	"""Compute a stable display range across multiple arrays."""

	values = [np.asarray(array, dtype=np.float32) for array in arrays]
	combined_min = min(float(np.nanmin(array)) for array in values)
	combined_max = max(float(np.nanmax(array)) for array in values)
	if not np.isfinite(combined_min) or not np.isfinite(combined_max):
		raise ValueError("Encountered non-finite values while computing display limits.")
	if np.isclose(combined_min, combined_max):
		combined_max = combined_min + 1.0
	return combined_min, combined_max


def _draw_contours(ax, ground_truth: np.ndarray, predicted: np.ndarray, threshold: float) -> None:
	"""Overlay ground-truth and predicted perimeter contours on an axis."""

	gt_min, gt_max = float(np.nanmin(ground_truth)), float(np.nanmax(ground_truth))
	pred_min, pred_max = float(np.nanmin(predicted)), float(np.nanmax(predicted))
	contour_drawn = False

	if gt_min <= threshold <= gt_max:
		ax.contour(ground_truth, levels=[threshold], colors=["cyan"], linewidths=1.8)
		contour_drawn = True
	if pred_min <= threshold <= pred_max:
		ax.contour(
			predicted,
			levels=[threshold],
			colors=["white"],
			linewidths=1.8,
			linestyles=["--"],
		)
		contour_drawn = True

	if contour_drawn:
		handles = [
			Line2D([0], [0], color="cyan", linewidth=2.0, label="Ground truth perimeter"),
			Line2D([0], [0], color="white", linewidth=2.0, linestyle="--", label="Predicted perimeter"),
		]
		ax.legend(handles=handles, loc="upper right", framealpha=0.85, fontsize=9)


def plot_prediction_grid(
	current_map,
	ground_truth_map,
	predicted_map,
	output_path: str | Path,
	title: str,
	threshold: float | None = None,
	cmap: str = "inferno",
	error_cmap: str = "magma",
	dpi: int = 150,
	normalization_stats: Mapping[str, np.ndarray] | None = None,
	channel_index: int = 0,
	draw_contours: bool = True,
) -> Path:
	"""Save a 5-panel visualization for one wildfire forecast sample.

	Panels:
	- current input map
	- ground-truth future map
	- predicted future map
	- absolute error map
	- predicted map with optional ground-truth/predicted perimeter contours
	"""

	current_map = inverse_normalize_channel_map(current_map, channel_index, normalization_stats)
	ground_truth_map = _as_2d_array(ground_truth_map)
	predicted_map = _as_2d_array(predicted_map)
	error_map = np.abs(predicted_map - ground_truth_map)
	shared_vmin, shared_vmax = _finite_min_max(current_map, ground_truth_map, predicted_map)
	error_vmax = max(float(np.nanmax(error_map)), 1e-6)

	fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=dpi, constrained_layout=True)
	axes_flat = axes.flatten()
	panel_specs = [
		("Current input fire intensity", current_map, cmap, shared_vmin, shared_vmax, False),
		("Ground truth future fire intensity", ground_truth_map, cmap, shared_vmin, shared_vmax, False),
		("Predicted future fire intensity", predicted_map, cmap, shared_vmin, shared_vmax, False),
		("Absolute error map", error_map, error_cmap, 0.0, error_vmax, False),
		("Contour overlay", predicted_map, cmap, shared_vmin, shared_vmax, True),
	]

	for axis, (panel_title, panel_data, panel_cmap, vmin, vmax, overlay_contours) in zip(axes_flat, panel_specs):
		image = axis.imshow(panel_data, origin="lower", cmap=panel_cmap, vmin=vmin, vmax=vmax)
		axis.set_title(panel_title)
		axis.set_xticks([])
		axis.set_yticks([])
		if overlay_contours and draw_contours and threshold is not None:
			_draw_contours(axis, ground_truth_map, predicted_map, threshold)
		fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

	for axis in axes_flat[len(panel_specs):]:
		axis.axis("off")

	fig.suptitle(title, fontsize=14)
	output_path = Path(output_path).expanduser().resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)
	return output_path
