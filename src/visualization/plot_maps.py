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


def plot_model_vs_persistence_grid(
	last_input_target_map,
	ground_truth_map,
	persistence_prediction_map,
	model_prediction_map,
	output_path: str | Path,
	title: str,
	cmap: str = "inferno",
	error_cmap: str = "magma",
	dpi: int = 150,
) -> Path:
	"""Save a 6-panel model-vs-persistence comparison figure."""

	last_input_target_map = _as_2d_array(last_input_target_map)
	ground_truth_map = _as_2d_array(ground_truth_map)
	persistence_prediction_map = _as_2d_array(persistence_prediction_map)
	model_prediction_map = _as_2d_array(model_prediction_map)

	persistence_error_map = np.abs(persistence_prediction_map - ground_truth_map)
	model_error_map = np.abs(model_prediction_map - ground_truth_map)
	shared_vmin, shared_vmax = _finite_min_max(
		last_input_target_map,
		ground_truth_map,
		persistence_prediction_map,
		model_prediction_map,
	)
	error_vmin, error_vmax = _finite_min_max(persistence_error_map, model_error_map)

	fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=dpi, constrained_layout=True)
	panel_specs = [
		("Last input target map", last_input_target_map, cmap, shared_vmin, shared_vmax),
		("Ground truth future target", ground_truth_map, cmap, shared_vmin, shared_vmax),
		("Persistence prediction", persistence_prediction_map, cmap, shared_vmin, shared_vmax),
		("Model prediction", model_prediction_map, cmap, shared_vmin, shared_vmax),
		("Persistence absolute error", persistence_error_map, error_cmap, error_vmin, error_vmax),
		("Model absolute error", model_error_map, error_cmap, error_vmin, error_vmax),
	]

	value_axes = []
	error_axes = []
	value_image = None
	error_image = None
	for axis, (panel_title, panel_data, panel_cmap, vmin, vmax) in zip(axes.flatten(), panel_specs):
		image = axis.imshow(panel_data, origin="lower", cmap=panel_cmap, vmin=vmin, vmax=vmax)
		axis.set_title(panel_title)
		axis.set_xticks([])
		axis.set_yticks([])
		if "error" in panel_title.lower():
			error_axes.append(axis)
			if error_image is None:
				error_image = image
		else:
			value_axes.append(axis)
			if value_image is None:
				value_image = image

	if value_image is not None and value_axes:
		fig.colorbar(value_image, ax=value_axes, fraction=0.03, pad=0.02)
	if error_image is not None and error_axes:
		fig.colorbar(error_image, ax=error_axes, fraction=0.03, pad=0.02)

	fig.suptitle(title, fontsize=14)
	output_path = Path(output_path).expanduser().resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)
	return output_path


def plot_multitask_prediction_grid(
	current_surface_fuel,
	true_future_surface_fuel,
	pred_future_surface_fuel,
	current_canopy_fuel,
	true_future_canopy_fuel,
	pred_future_canopy_fuel,
	true_mask,
	pred_mask_probability,
	pred_mask,
	output_path: str | Path,
	title: str,
	cmap: str = "inferno",
	error_cmap: str = "magma",
	mask_cmap: str = "viridis",
	dpi: int = 150,
) -> Path:
	"""Save a 12-panel visualization for one multitask wildfire forecast sample."""

	current_surface_fuel = _as_2d_array(current_surface_fuel)
	true_future_surface_fuel = _as_2d_array(true_future_surface_fuel)
	pred_future_surface_fuel = _as_2d_array(pred_future_surface_fuel)
	current_canopy_fuel = _as_2d_array(current_canopy_fuel)
	true_future_canopy_fuel = _as_2d_array(true_future_canopy_fuel)
	pred_future_canopy_fuel = _as_2d_array(pred_future_canopy_fuel)
	true_mask = _as_2d_array(true_mask)
	pred_mask_probability = _as_2d_array(pred_mask_probability)
	pred_mask = _as_2d_array(pred_mask)

	surface_error = np.abs(pred_future_surface_fuel - true_future_surface_fuel)
	canopy_error = np.abs(pred_future_canopy_fuel - true_future_canopy_fuel)
	surface_vmin, surface_vmax = _finite_min_max(current_surface_fuel, true_future_surface_fuel, pred_future_surface_fuel)
	canopy_vmin, canopy_vmax = _finite_min_max(current_canopy_fuel, true_future_canopy_fuel, pred_future_canopy_fuel)
	error_vmin, error_vmax = _finite_min_max(surface_error, canopy_error)

	fig, axes = plt.subplots(4, 3, figsize=(18, 20), dpi=dpi, constrained_layout=True)
	panel_specs = [
		("Current surface fuel", current_surface_fuel, cmap, surface_vmin, surface_vmax),
		("True future surface fuel", true_future_surface_fuel, cmap, surface_vmin, surface_vmax),
		("Predicted future surface fuel", pred_future_surface_fuel, cmap, surface_vmin, surface_vmax),
		("Surface fuel prediction error", surface_error, error_cmap, error_vmin, error_vmax),
		("Current canopy fuel", current_canopy_fuel, cmap, canopy_vmin, canopy_vmax),
		("True future canopy fuel", true_future_canopy_fuel, cmap, canopy_vmin, canopy_vmax),
		("Predicted future canopy fuel", pred_future_canopy_fuel, cmap, canopy_vmin, canopy_vmax),
		("Canopy fuel prediction error", canopy_error, error_cmap, error_vmin, error_vmax),
		("True mask", true_mask, mask_cmap, 0.0, 1.0),
		("Predicted mask probability", pred_mask_probability, mask_cmap, 0.0, 1.0),
		("Predicted binary mask", pred_mask, mask_cmap, 0.0, 1.0),
		("Predicted / true perimeter contours", pred_mask_probability, mask_cmap, 0.0, 1.0),
	]

	for axis, (panel_title, panel_data, panel_cmap, vmin, vmax) in zip(axes.flatten(), panel_specs):
		image = axis.imshow(panel_data, origin="lower", cmap=panel_cmap, vmin=vmin, vmax=vmax)
		axis.set_title(panel_title)
		axis.set_xticks([])
		axis.set_yticks([])
		if panel_title == "Predicted / true perimeter contours":
			_draw_contours(axis, true_mask, pred_mask_probability, threshold=0.5)
		fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

	fig.suptitle(title, fontsize=14)
	output_path = Path(output_path).expanduser().resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)
	return output_path


def plot_reconstructed_fuel_beds_grid(
	current_surface_fuel,
	true_future_surface_fuel,
	pred_future_surface_fuel,
	current_canopy_fuel,
	true_future_canopy_fuel,
	pred_future_canopy_fuel,
	output_path: str | Path,
	title: str,
	cmap: str = "inferno",
	error_cmap: str = "magma",
	dpi: int = 150,
) -> Path:
	"""Save an 8-panel reconstructed-fuel-bed comparison figure."""

	current_surface_fuel = _as_2d_array(current_surface_fuel)
	true_future_surface_fuel = _as_2d_array(true_future_surface_fuel)
	pred_future_surface_fuel = _as_2d_array(pred_future_surface_fuel)
	current_canopy_fuel = _as_2d_array(current_canopy_fuel)
	true_future_canopy_fuel = _as_2d_array(true_future_canopy_fuel)
	pred_future_canopy_fuel = _as_2d_array(pred_future_canopy_fuel)
	surface_error = np.abs(pred_future_surface_fuel - true_future_surface_fuel)
	canopy_error = np.abs(pred_future_canopy_fuel - true_future_canopy_fuel)

	surface_vmin, surface_vmax = _finite_min_max(current_surface_fuel, true_future_surface_fuel, pred_future_surface_fuel)
	canopy_vmin, canopy_vmax = _finite_min_max(current_canopy_fuel, true_future_canopy_fuel, pred_future_canopy_fuel)
	error_vmin, error_vmax = _finite_min_max(surface_error, canopy_error)

	fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=dpi, constrained_layout=True)
	panel_specs = [
		("Current surface fuel", current_surface_fuel, cmap, surface_vmin, surface_vmax),
		("True future surface fuel", true_future_surface_fuel, cmap, surface_vmin, surface_vmax),
		("Predicted future surface fuel", pred_future_surface_fuel, cmap, surface_vmin, surface_vmax),
		("Surface fuel error", surface_error, error_cmap, error_vmin, error_vmax),
		("Current canopy fuel", current_canopy_fuel, cmap, canopy_vmin, canopy_vmax),
		("True future canopy fuel", true_future_canopy_fuel, cmap, canopy_vmin, canopy_vmax),
		("Predicted future canopy fuel", pred_future_canopy_fuel, cmap, canopy_vmin, canopy_vmax),
		("Canopy fuel error", canopy_error, error_cmap, error_vmin, error_vmax),
	]
	for axis, (panel_title, panel_data, panel_cmap, vmin, vmax) in zip(axes.flatten(), panel_specs):
		image = axis.imshow(panel_data, origin="lower", cmap=panel_cmap, vmin=vmin, vmax=vmax)
		axis.set_title(panel_title)
		axis.set_xticks([])
		axis.set_yticks([])
		fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

	fig.suptitle(title, fontsize=14)
	output_path = Path(output_path).expanduser().resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)
	return output_path
