"""Animation helpers for autoregressive wildfire rollout visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
import numpy as np


def _as_2d_array(array) -> np.ndarray:
	"""Convert a tensor-like object into a 2D NumPy array."""

	if hasattr(array, "detach"):
		array = array.detach().cpu().numpy()
	else:
		array = np.asarray(array)

	if array.ndim == 3 and array.shape[0] == 1:
		array = array[0]
	if array.ndim != 2:
		raise ValueError(f"Expected a 2D map, got shape {array.shape}.")
	return np.asarray(array, dtype=np.float32)


def _finite_min_max(*arrays: np.ndarray | None) -> tuple[float, float]:
	"""Compute a robust visualization range across one or more arrays."""

	values = [np.asarray(array, dtype=np.float32) for array in arrays if array is not None]
	if not values:
		return 0.0, 1.0
	combined_min = min(float(np.nanmin(array)) for array in values)
	combined_max = max(float(np.nanmax(array)) for array in values)
	if not np.isfinite(combined_min) or not np.isfinite(combined_max):
		raise ValueError("Encountered non-finite values while computing display limits.")
	if np.isclose(combined_min, combined_max):
		combined_max = combined_min + 1.0
	return combined_min, combined_max


def save_single_map(
	map_array,
	output_path: str | Path,
	title: str,
	cmap: str = "inferno",
	dpi: int = 150,
	vmin: float | None = None,
	vmax: float | None = None,
) -> Path:
	"""Save a single top-view map with a title and colorbar."""

	map_2d = _as_2d_array(map_array)
	if vmin is None or vmax is None:
		vmin, vmax = _finite_min_max(map_2d)

	fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi, constrained_layout=True)
	image = ax.imshow(map_2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
	ax.set_title(title)
	ax.set_xticks([])
	ax.set_yticks([])
	fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

	output_path = Path(output_path).expanduser().resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)
	return output_path


def save_side_by_side_maps(
	left_map,
	right_map,
	output_path: str | Path,
	left_title: str,
	right_title: str,
	title: str,
	cmap: str = "inferno",
	dpi: int = 150,
	right_available: bool = True,
) -> Path:
	"""Save a side-by-side map comparison with colorbars."""

	left_2d = _as_2d_array(left_map)
	right_2d = _as_2d_array(right_map) if right_available and right_map is not None else None
	vmin, vmax = _finite_min_max(left_2d, right_2d)

	fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi, constrained_layout=True)
	left_image = axes[0].imshow(left_2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
	axes[0].set_title(left_title)
	axes[0].set_xticks([])
	axes[0].set_yticks([])
	fig.colorbar(left_image, ax=axes[0], fraction=0.046, pad=0.04)

	if right_2d is None:
		axes[1].text(0.5, 0.5, "Ground truth unavailable", ha="center", va="center", fontsize=12)
		axes[1].set_title(right_title)
		axes[1].set_xticks([])
		axes[1].set_yticks([])
		axes[1].set_frame_on(False)
	else:
		right_image = axes[1].imshow(right_2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
		axes[1].set_title(right_title)
		axes[1].set_xticks([])
		axes[1].set_yticks([])
		fig.colorbar(right_image, ax=axes[1], fraction=0.046, pad=0.04)

	fig.suptitle(title, fontsize=14)
	output_path = Path(output_path).expanduser().resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)
	return output_path


def save_rollout_animation(
	predicted_frames: Sequence[np.ndarray],
	output_path: str | Path,
	titles: Sequence[str] | None = None,
	ground_truth_frames: Sequence[np.ndarray | None] | None = None,
	cmap: str = "inferno",
	fps: int = 2,
) -> Path:
	"""Save an animation comparing predicted rollout frames against ground truth when available."""

	if not predicted_frames:
		raise ValueError("predicted_frames must contain at least one frame.")
	if ground_truth_frames is not None and len(ground_truth_frames) != len(predicted_frames):
		raise ValueError("ground_truth_frames must match predicted_frames length when provided.")
	if titles is not None and len(titles) != len(predicted_frames):
		raise ValueError("titles must match predicted_frames length when provided.")

	predicted_arrays = [_as_2d_array(frame) for frame in predicted_frames]
	ground_truth_arrays = None
	if ground_truth_frames is not None:
		ground_truth_arrays = [None if frame is None else _as_2d_array(frame) for frame in ground_truth_frames]

	finite_ground_truth = [frame for frame in ground_truth_arrays or [] if frame is not None]
	vmin, vmax = _finite_min_max(*(predicted_arrays + finite_ground_truth))

	fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
	pred_image = axes[0].imshow(predicted_arrays[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
	axes[0].set_title("Predicted")
	axes[0].set_xticks([])
	axes[0].set_yticks([])
	fig.colorbar(pred_image, ax=axes[0], fraction=0.046, pad=0.04)

	if ground_truth_arrays is not None and ground_truth_arrays[0] is not None:
		gt_image = axes[1].imshow(ground_truth_arrays[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
		fig.colorbar(gt_image, ax=axes[1], fraction=0.046, pad=0.04)
	else:
		axes[1].text(0.5, 0.5, "Ground truth unavailable", ha="center", va="center", fontsize=12)
	axes[1].set_title("Ground truth")
	axes[1].set_xticks([])
	axes[1].set_yticks([])

	def _update(frame_index: int):
		pred_image.set_data(predicted_arrays[frame_index])
		axes[1].cla()
		axes[1].set_title("Ground truth")
		axes[1].set_xticks([])
		axes[1].set_yticks([])
		if ground_truth_arrays is None or ground_truth_arrays[frame_index] is None:
			axes[1].text(0.5, 0.5, "Ground truth unavailable", ha="center", va="center", fontsize=12)
		else:
			axes[1].imshow(ground_truth_arrays[frame_index], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
		if titles is not None:
			fig.suptitle(titles[frame_index], fontsize=14)
		return (pred_image,)

	animation = FuncAnimation(fig, _update, frames=len(predicted_arrays), interval=max(1, int(1000 / max(1, fps))), blit=False)

	output_path = Path(output_path).expanduser().resolve()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if output_path.suffix.lower() == ".mp4":
		animation.save(output_path, writer=FFMpegWriter(fps=max(1, fps)))
	elif output_path.suffix.lower() == ".gif":
		animation.save(output_path, writer=PillowWriter(fps=max(1, fps)))
	else:
		if writers.is_available("ffmpeg"):
			output_path = output_path.with_suffix(".mp4")
			animation.save(output_path, writer=FFMpegWriter(fps=max(1, fps)))
		else:
			output_path = output_path.with_suffix(".gif")
			animation.save(output_path, writer=PillowWriter(fps=max(1, fps)))

	plt.close(fig)
	return output_path
