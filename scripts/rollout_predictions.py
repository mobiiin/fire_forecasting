"""Generate multitask autoregressive rollout GIFs for the wildfire ConvLSTM U-Net."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

try:
	import imageio.v2 as imageio  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - dependency-specific fallback
	imageio = None

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None

from src.config import load_config
from src.data.dataset import (
	FireSequenceDataset,
	MultiFireSequenceDataset,
	_resolve_multitask_config,
	build_model_input_from_raw_window,
	create_dataloaders,
)
from src.data.spatial_transforms import infer_with_external_test_spatial_handling
from src.models.convlstm_unet import build_model_from_config
from src.training.checkpoints import latest_and_best_checkpoint_paths, load_checkpoint
from src.training.train import _ensure_config_path, _get_device
from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.visualization.fuel_reconstruction import reconstruct_future_fuel_bed

EPS = 1e-8


def _get_section(config: Mapping[str, Any], *names: str) -> dict[str, Any]:
	"""Return the first nested mapping found under any of the provided names."""

	for name in names:
		section = config.get(name)
		if isinstance(section, Mapping):
			return dict(section)
	return {}


def _resolve_path(base_path: Path | None, configured_path: str | Path) -> Path:
	"""Resolve a configured path relative to the config file when needed."""

	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	if base_path is None:
		return path.resolve()
	return (base_path.parent / path).resolve()


def _resolve_rollout_config(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Resolve rollout-specific settings with backward-compatible fallbacks."""

	section = _get_section(config, "rollout")
	training_section = _get_section(config, "training")
	return {
		"enabled": bool(section.get("enabled", True)),
		"rollout_steps": int(section.get("rollout_steps", 20)),
		"window_mode": str(section.get("window_mode", "static")).lower(),
		"exogenous_mode": str(section.get("exogenous_mode", "teacher_forced")).lower(),
		"surface_fuel_channel": int(section.get("surface_fuel_channel", 84)),
		"canopy_fuel_channel": int(section.get("canopy_fuel_channel", 85)),
		"clamp_fuel_nonnegative": bool(section.get("clamp_fuel_nonnegative", True)),
		"mask_probability_threshold": float(section.get("mask_probability_threshold", 0.5)),
		"save_gif": bool(section.get("save_gif", section.get("save_animation", True))),
		"save_png_frames": bool(section.get("save_png_frames", section.get("save_step_figures", False))),
		"fps": int(section.get("fps", 2)),
		"output_dir": str(section.get("output_dir", "./outputs/rollouts")),
		"random_seed": int(section.get("random_seed", training_section.get("seed", 42))),
	}


def _build_dataset_for_split(config: Mapping[str, Any], split: str):
	"""Build the dataset backing one requested split."""

	config = dict(config)
	config["return_metadata"] = True
	train_loader, val_loader, test_loader = create_dataloaders(config)
	split = str(split).lower()
	loader_by_split = {"train": train_loader, "val": val_loader, "test": test_loader}
	if split not in loader_by_split:
		raise ValueError(f"split must be one of 'train', 'val', or 'test', got {split!r}.")
	selected_loader = loader_by_split[split]
	if selected_loader is None:
		raise ValueError(
			f"Requested split {split!r} is not available for split_mode={config.get('split_mode', 'train_val_test')!r}."
		)
	return selected_loader.dataset


def _resolve_checkpoint_path(config: Mapping[str, Any]) -> Path:
	"""Resolve the best checkpoint path, falling back to the latest checkpoint."""

	checkpoint_config = _get_section(config, "checkpoint")
	checkpoint_path = checkpoint_config.get("path", "./artifacts/checkpoints/convlstm_unet.pt")
	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	latest_path, best_path = latest_and_best_checkpoint_paths(_resolve_path(config_path, checkpoint_path))
	selected = best_path if best_path.exists() else latest_path
	if not selected.exists():
		raise FileNotFoundError(
			"No checkpoint found for rollout evaluation. "
			f"Checked best='{best_path}' and latest='{latest_path}'."
		)
	return selected


def _denormalize_predicted_consumed_channels(dataset, y_pred: torch.Tensor) -> torch.Tensor:
	"""Undo optional multitask target normalization for the two regression heads only."""

	if not bool(getattr(dataset, "normalize_target", False)):
		return y_pred
	target_mean = getattr(dataset, "target_mean", None)
	target_std = getattr(dataset, "target_std", None)
	if target_mean is None or target_std is None:
		return y_pred
	mean_tensor = torch.as_tensor(target_mean, dtype=y_pred.dtype, device=y_pred.device).reshape(1, -1, 1, 1)
	std_tensor = torch.as_tensor(target_std, dtype=y_pred.dtype, device=y_pred.device).reshape(1, -1, 1, 1)
	std_tensor = torch.clamp(std_tensor, min=1e-6)
	y_pred = y_pred.clone()
	regression_channels = min(int(mean_tensor.shape[1]), 2)
	y_pred[:, :regression_channels] = y_pred[:, :regression_channels] * std_tensor[:, :regression_channels] + mean_tensor[:, :regression_channels]
	return y_pred


def _load_raw_frame(file_path: str | Path) -> np.ndarray:
	"""Load one raw dataset frame as float32."""

	array = np.load(Path(file_path).expanduser().resolve(), mmap_mode="r", allow_pickle=False)
	if array.ndim != 3:
		raise ValueError(f"Expected a 3D raw frame, got shape {array.shape}.")
	return np.asarray(array, dtype=np.float32)


def _dataset_context(dataset, split_sample_index: int) -> dict[str, Any]:
	"""Resolve raw-file context for one split sample from either dataset type."""

	if split_sample_index < 0 or split_sample_index >= len(dataset):
		raise IndexError(
			f"sample index must be within [0, {max(0, len(dataset) - 1)}], got {split_sample_index}."
		)

	if isinstance(dataset, MultiFireSequenceDataset):
		ref = dataset.sample_refs[split_sample_index]
		dataset_id = int(ref["dataset_id"])
		record = dataset.dataset_records[dataset_id]
		file_paths = list(record["file_paths"])
		sample_start = int(ref["sample_index"])
		initial_fuel = dataset.initial_fuel_maps.get(dataset_id)
		dataset_name = str(record["dataset_name"])
	elif isinstance(dataset, FireSequenceDataset):
		file_paths = list(dataset.file_paths)
		sample_start = int(dataset.sample_indices[split_sample_index])
		initial_fuel = getattr(dataset, "initial_fuel_map", None)
		dataset_name = Path(file_paths[0]).parent.name if file_paths else "dataset"
	else:
		raise TypeError(f"Unsupported dataset type for rollout: {type(dataset)!r}.")

	if initial_fuel is None:
		raise ValueError("Rollout requires initial fuel to be available for the selected dataset.")

	current_index = sample_start + int(dataset.input_sequence_length) - 1
	return {
		"dataset_name": dataset_name,
		"file_paths": file_paths,
		"sample_start": sample_start,
		"current_index": current_index,
		"initial_fuel": np.asarray(initial_fuel, dtype=np.float32),
		"split_sample_index": int(split_sample_index),
	}


def _mask_from_raw_frame(
	frame: np.ndarray,
	initial_fuel: np.ndarray,
	config: Mapping[str, Any],
	previous_true_surface_fuel: np.ndarray | None = None,
	previous_true_canopy_fuel: np.ndarray | None = None,
) -> np.ndarray:
	"""Construct the configured true mask directly from one raw frame."""

	multitask = _resolve_multitask_config(config)
	mask_target_type = str(multitask["mask_target_type"]).lower()
	if mask_target_type == "active_flux":
		flux_channel = int(multitask["flux_mask_channel"])
		mask = frame[:, :, flux_channel] > float(multitask["flux_fire_threshold"])
	elif mask_target_type == "burned_fuel":
		surface_channel = int(multitask["surface_fuel_channel"])
		canopy_channel = int(multitask["canopy_fuel_channel"])
		surface_consumed = np.asarray(initial_fuel[:, :, 0], dtype=np.float32) - np.asarray(frame[:, :, surface_channel], dtype=np.float32)
		canopy_consumed = np.asarray(initial_fuel[:, :, 1], dtype=np.float32) - np.asarray(frame[:, :, canopy_channel], dtype=np.float32)
		if bool(multitask["clamp_consumed_fuel_targets_nonnegative"]):
			surface_consumed = np.maximum(surface_consumed, 0.0)
			canopy_consumed = np.maximum(canopy_consumed, 0.0)
		mask = np.maximum(surface_consumed, canopy_consumed) > float(multitask["consumed_fuel_threshold"])
	elif mask_target_type == "step_consumed_fuel":
		if previous_true_surface_fuel is None or previous_true_canopy_fuel is None:
			raise ValueError(
				"mask_target_type='step_consumed_fuel' requires previous_true_surface_fuel and previous_true_canopy_fuel."
			)
		surface_channel = int(multitask["surface_fuel_channel"])
		canopy_channel = int(multitask["canopy_fuel_channel"])
		surface_consumed = np.asarray(previous_true_surface_fuel, dtype=np.float32) - np.asarray(frame[:, :, surface_channel], dtype=np.float32)
		canopy_consumed = np.asarray(previous_true_canopy_fuel, dtype=np.float32) - np.asarray(frame[:, :, canopy_channel], dtype=np.float32)
		if bool(multitask["clamp_consumed_fuel_targets_nonnegative"]):
			surface_consumed = np.maximum(surface_consumed, 0.0)
			canopy_consumed = np.maximum(canopy_consumed, 0.0)
		mask = np.maximum(surface_consumed, canopy_consumed) > float(multitask["consumed_fuel_threshold"])
	else:
		raise ValueError(
			"Unsupported multitask.mask_target_type for rollout mask construction. "
			f"Expected 'active_flux', 'burned_fuel', or 'step_consumed_fuel', got {mask_target_type!r}."
		)
	return np.asarray(mask, dtype=np.float32)


def _segmentation_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
	"""Compute binary segmentation metrics for one pair of masks."""

	pred_bool = np.asarray(pred_mask, dtype=bool)
	true_bool = np.asarray(true_mask, dtype=bool)
	tp = float(np.logical_and(pred_bool, true_bool).sum())
	fp = float(np.logical_and(pred_bool, np.logical_not(true_bool)).sum())
	fn = float(np.logical_and(np.logical_not(pred_bool), true_bool).sum())
	tn = float(np.logical_and(np.logical_not(pred_bool), np.logical_not(true_bool)).sum())

	if tp + fp + fn <= EPS and tn >= 0.0:
		iou = 1.0
		dice = 1.0
		precision = 1.0
		recall = 1.0
	else:
		iou = tp / (tp + fp + fn + EPS)
		dice = (2.0 * tp) / (2.0 * tp + fp + fn + EPS)
		precision = tp / (tp + fp + EPS)
		recall = tp / (tp + fn + EPS)

	return {
		"mask_iou": float(iou),
		"mask_dice": float(dice),
		"mask_precision": float(precision),
		"mask_recall": float(recall),
		"predicted_active_fraction": float(pred_bool.mean()),
		"true_active_fraction": float(true_bool.mean()),
	}


def _write_metrics_csv(output_path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
	"""Write rollout metrics rows to CSV."""

	output_path.parent.mkdir(parents=True, exist_ok=True)
	if not rows:
		output_path.write_text("", encoding="utf-8")
		return output_path

	fieldnames: list[str] = []
	seen = set()
	for row in rows:
		for key in row.keys():
			if key not in seen:
				fieldnames.append(key)
				seen.add(key)
	with output_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)
	return output_path


def _predict_one_step(model, dataset, config: Mapping[str, Any], device, model_input_tensor: torch.Tensor, split: str) -> tuple[np.ndarray, str]:
	"""Run one forward pass and return denormalized multitask outputs on the native grid."""

	use_external_spatial = str(config.get("split_mode", "train_val_test")).lower() == "train_val_external_test" and str(split).lower() == "test"
	with torch.no_grad():
		if use_external_spatial:
			inference_result = infer_with_external_test_spatial_handling(model, model_input_tensor, config)
			y_pred = inference_result["y_pred"]
			mode_used = str(inference_result["mode_used"])
		else:
			y_pred = model(model_input_tensor)
			mode_used = "direct"
		y_pred = _denormalize_predicted_consumed_channels(dataset, y_pred)

	if tuple(y_pred.shape[:2]) != (1, 3):
		raise ValueError(f"Expected model rollout output shape (1, 3, H, W), got {tuple(y_pred.shape)}.")
	return y_pred.detach().cpu().numpy()[0].astype(np.float32, copy=False), mode_used


def _mae(prediction: np.ndarray, target: np.ndarray) -> float:
	return float(np.mean(np.abs(np.asarray(prediction, dtype=np.float32) - np.asarray(target, dtype=np.float32))))


def _safe_min_max(arrays: Sequence[np.ndarray]) -> tuple[float, float]:
	"""Compute a stable visualization range."""

	vmin = min(float(np.nanmin(np.asarray(array, dtype=np.float32))) for array in arrays)
	vmax = max(float(np.nanmax(np.asarray(array, dtype=np.float32))) for array in arrays)
	if np.isclose(vmin, vmax):
		vmax = vmin + 1.0
	return vmin, vmax


def _render_rollout_frame(
	step_record: Mapping[str, Any],
	surface_limits: tuple[float, float],
	canopy_limits: tuple[float, float],
	error_limits: tuple[float, float],
) -> np.ndarray:
	"""Render one GIF frame as a 3x3 comparison grid and return RGBA pixels."""

	true_surface = np.asarray(step_record["true_surface_fuel"], dtype=np.float32)
	pred_surface = np.asarray(step_record["pred_surface_fuel"], dtype=np.float32)
	true_canopy = np.asarray(step_record["true_canopy_fuel"], dtype=np.float32)
	pred_canopy = np.asarray(step_record["pred_canopy_fuel"], dtype=np.float32)
	surface_error = np.abs(pred_surface - true_surface)
	canopy_error = np.abs(pred_canopy - true_canopy)
	true_mask = np.asarray(step_record["true_mask"], dtype=np.float32)
	pred_mask_probability = np.asarray(step_record["pred_mask_probability"], dtype=np.float32)
	pred_mask = np.asarray(step_record["pred_mask"], dtype=np.float32)

	fig, axes = plt.subplots(3, 3, figsize=(14, 12), dpi=120, constrained_layout=True)
	panel_specs = [
		("Ground truth surface fuel", true_surface, "inferno", surface_limits[0], surface_limits[1]),
		("Predicted surface fuel", pred_surface, "inferno", surface_limits[0], surface_limits[1]),
		("Surface fuel absolute error", surface_error, "magma", error_limits[0], error_limits[1]),
		("Ground truth canopy fuel", true_canopy, "inferno", canopy_limits[0], canopy_limits[1]),
		("Predicted canopy fuel", pred_canopy, "inferno", canopy_limits[0], canopy_limits[1]),
		("Canopy fuel absolute error", canopy_error, "magma", error_limits[0], error_limits[1]),
		("Ground truth mask", true_mask, "viridis", 0.0, 1.0),
		("Predicted mask probability", pred_mask_probability, "viridis", 0.0, 1.0),
		("Perimeter overlay", pred_mask_probability, "viridis", 0.0, 1.0),
	]

	for axis, (title, panel_data, cmap, vmin, vmax) in zip(axes.flatten(), panel_specs, strict=True):
		image = axis.imshow(panel_data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
		axis.set_title(title, fontsize=10)
		axis.set_xticks([])
		axis.set_yticks([])
		if title == "Perimeter overlay":
			if np.any(true_mask > 0.5):
				axis.contour(true_mask, levels=[0.5], colors=["cyan"], linewidths=1.8)
			if np.any(pred_mask > 0.5):
				axis.contour(pred_mask, levels=[0.5], colors=["white"], linewidths=1.8, linestyles=["--"])
		fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

	fig.suptitle(str(step_record["title"]), fontsize=14)
	fig.canvas.draw()
	frame_rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
	plt.close(fig)
	return frame_rgba


def _save_rollout_gif(
	step_records: Sequence[Mapping[str, Any]],
	output_path: Path,
	fps: int,
	save_png_frames: bool,
) -> Path:
	"""Render rollout frames and save them as one GIF."""

	if imageio is None:
		raise ImportError("rollout_predictions.py requires imageio. Install it with `pip install imageio`.")
	if not step_records:
		raise ValueError("Cannot create a rollout GIF without any rollout steps.")

	surface_arrays: list[np.ndarray] = []
	canopy_arrays: list[np.ndarray] = []
	error_arrays: list[np.ndarray] = []
	for record in step_records:
		true_surface = np.asarray(record["true_surface_fuel"], dtype=np.float32)
		pred_surface = np.asarray(record["pred_surface_fuel"], dtype=np.float32)
		true_canopy = np.asarray(record["true_canopy_fuel"], dtype=np.float32)
		pred_canopy = np.asarray(record["pred_canopy_fuel"], dtype=np.float32)
		surface_arrays.extend([true_surface, pred_surface])
		canopy_arrays.extend([true_canopy, pred_canopy])
		error_arrays.extend([np.abs(pred_surface - true_surface), np.abs(pred_canopy - true_canopy)])

	surface_limits = _safe_min_max(surface_arrays)
	canopy_limits = _safe_min_max(canopy_arrays)
	error_values = np.concatenate([array.reshape(-1) for array in error_arrays], axis=0)
	error_vmax = float(np.quantile(error_values, 0.99)) if error_values.size > 0 else 1.0
	error_vmax = max(error_vmax, 1e-6)
	error_limits = (0.0, error_vmax)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	frame_dir = output_path.parent / f"frames_sample{int(step_records[0]['split_sample_index']):05d}"
	gif_frames: list[np.ndarray] = []
	for frame_index, record in enumerate(step_records, start=1):
		frame_rgba = _render_rollout_frame(record, surface_limits, canopy_limits, error_limits)
		gif_frames.append(frame_rgba)
		if save_png_frames:
			frame_dir.mkdir(parents=True, exist_ok=True)
			imageio.imwrite(frame_dir / f"step_{frame_index:03d}.png", frame_rgba)

	imageio.mimsave(output_path, gif_frames, fps=max(1, int(fps)))
	return output_path


def _run_single_rollout(
	model,
	dataset,
	config: Mapping[str, Any],
	split: str,
	split_sample_index: int,
	rollout_steps: int,
	exogenous_mode: str,
	device,
	output_root: Path,
	fps: int,
	save_png_frames: bool,
) -> tuple[Path | None, list[dict[str, Any]]]:
	"""Run one autoregressive rollout and save the comparison GIF."""

	rollout_config = _resolve_rollout_config(config)
	if rollout_config["window_mode"] != "static":
		raise NotImplementedError(
			f"rollout.window_mode={rollout_config['window_mode']!r} is not implemented yet. Only 'static' is supported."
		)
	if exogenous_mode not in {"teacher_forced", "constant"}:
		raise ValueError(f"exogenous_mode must be 'teacher_forced' or 'constant', got {exogenous_mode!r}.")
	if int(dataset.prediction_horizon) != 1:
		raise NotImplementedError(
			"Autoregressive rollout currently requires prediction_horizon == 1 because each step advances the raw window by one frame. "
			f"The configured/trained horizon is {int(dataset.prediction_horizon)}; use direct evaluation for this H-step target."
		)

	context = _dataset_context(dataset, split_sample_index)
	file_paths = context["file_paths"]
	sample_start = int(context["sample_start"])
	current_index = int(context["current_index"])
	initial_fuel = np.asarray(context["initial_fuel"], dtype=np.float32)
	dataset_name = str(context["dataset_name"])

	available_truth_steps = max(0, len(file_paths) - current_index - 1)
	if available_truth_steps <= 0:
		raise ValueError(f"No future ground-truth frames are available for sample index {split_sample_index}.")
	rollout_limit = min(int(rollout_steps), available_truth_steps)
	if rollout_limit < int(rollout_steps):
		print(
			f"Requested rollout_steps={rollout_steps} but only {available_truth_steps} future frames are available for sample index {split_sample_index}. "
			f"Using rollout_steps={rollout_limit}."
		)

	raw_window = np.stack(
		[_load_raw_frame(file_paths[index]) for index in range(sample_start, sample_start + int(dataset.input_sequence_length))],
		axis=0,
	).astype(np.float32, copy=False)
	previous_frame_before_window = _load_raw_frame(file_paths[sample_start - 1]) if sample_start > 0 else raw_window[0].copy()

	surface_channel = int(rollout_config["surface_fuel_channel"])
	canopy_channel = int(rollout_config["canopy_fuel_channel"])
	mask_threshold = float(rollout_config["mask_probability_threshold"])
	initial_true_frame = raw_window[-1].copy()
	initial_true_surface_fuel = np.asarray(initial_true_frame[:, :, surface_channel], dtype=np.float32)
	initial_true_canopy_fuel = np.asarray(initial_true_frame[:, :, canopy_channel], dtype=np.float32)

	previous_true_surface_fuel = initial_true_surface_fuel.copy()
	previous_true_canopy_fuel = initial_true_canopy_fuel.copy()
	step_records: list[dict[str, Any]] = []
	metric_rows: list[dict[str, Any]] = []
	mode_used_history: list[str] = []

	for rollout_step in range(1, rollout_limit + 1):
		model_input_np = build_model_input_from_raw_window(
			raw_window=raw_window,
			config=config,
			normalization_stats=getattr(dataset, "normalization_stats", None),
			initial_fuel=initial_fuel,
			previous_frame_before_window=previous_frame_before_window,
		)
		model_input_tensor = torch.from_numpy(model_input_np).to(device=device, dtype=torch.float32)
		y_pred, mode_used = _predict_one_step(model, dataset, config, device, model_input_tensor, split)
		mode_used_history.append(mode_used)

		last_rollout_frame = raw_window[-1]
		current_surface_fuel = np.asarray(last_rollout_frame[:, :, surface_channel], dtype=np.float32)
		current_canopy_fuel = np.asarray(last_rollout_frame[:, :, canopy_channel], dtype=np.float32)
		pred_surface_consumed = np.asarray(y_pred[0], dtype=np.float32)
		pred_canopy_consumed = np.asarray(y_pred[1], dtype=np.float32)
		pred_mask_logits = np.asarray(y_pred[2], dtype=np.float32)
		pred_mask_probability = 1.0 / (1.0 + np.exp(-np.clip(pred_mask_logits, -60.0, 60.0)))
		pred_mask = (pred_mask_probability > mask_threshold).astype(np.float32, copy=False)

		pred_surface_fuel, pred_canopy_fuel = reconstruct_future_fuel_bed(
			current_surface_fuel=current_surface_fuel,
			current_canopy_fuel=current_canopy_fuel,
			pred_surface_consumed=pred_surface_consumed,
			pred_canopy_consumed=pred_canopy_consumed,
			clamp_nonnegative=bool(rollout_config["clamp_fuel_nonnegative"]),
		)
		pred_surface_fuel = np.asarray(pred_surface_fuel, dtype=np.float32)
		pred_canopy_fuel = np.asarray(pred_canopy_fuel, dtype=np.float32)

		true_frame_index = current_index + rollout_step
		true_frame = _load_raw_frame(file_paths[true_frame_index])
		true_surface_fuel = np.asarray(true_frame[:, :, surface_channel], dtype=np.float32)
		true_canopy_fuel = np.asarray(true_frame[:, :, canopy_channel], dtype=np.float32)
		true_mask = _mask_from_raw_frame(
			frame=true_frame,
			initial_fuel=initial_fuel,
			config=config,
			previous_true_surface_fuel=previous_true_surface_fuel,
			previous_true_canopy_fuel=previous_true_canopy_fuel,
		)

		step_title = (
			f"{dataset_name} | sample {split_sample_index} | step {rollout_step:02d}/{rollout_limit:02d} "
			f"| target_idx {true_frame_index} | exogenous={exogenous_mode}"
		)
		step_record = {
			"dataset_name": dataset_name,
			"split": str(split),
			"split_sample_index": int(split_sample_index),
			"sample_start_index": int(sample_start),
			"rollout_step": int(rollout_step),
			"future_frame_index": int(true_frame_index),
			"title": step_title,
			"true_surface_fuel": true_surface_fuel,
			"pred_surface_fuel": pred_surface_fuel,
			"true_canopy_fuel": true_canopy_fuel,
			"pred_canopy_fuel": pred_canopy_fuel,
			"true_mask": true_mask,
			"pred_mask_probability": pred_mask_probability.astype(np.float32, copy=False),
			"pred_mask": pred_mask,
		}
		step_records.append(step_record)

		mask_metrics = _segmentation_metrics(pred_mask, true_mask)
		metric_rows.append(
			{
				"split": str(split),
				"dataset_name": dataset_name,
				"split_sample_index": int(split_sample_index),
				"sample_start_index": int(sample_start),
				"rollout_step": int(rollout_step),
				"future_frame_index": int(true_frame_index),
				"exogenous_mode": str(exogenous_mode),
				"inference_mode_used": str(mode_used),
				"surface_fuel_mae_model": _mae(pred_surface_fuel, true_surface_fuel),
				"canopy_fuel_mae_model": _mae(pred_canopy_fuel, true_canopy_fuel),
				**mask_metrics,
			}
		)

		if exogenous_mode == "teacher_forced":
			next_raw_frame = true_frame.copy()
		else:
			next_raw_frame = raw_window[-1].copy()
		next_raw_frame[:, :, surface_channel] = pred_surface_fuel
		next_raw_frame[:, :, canopy_channel] = pred_canopy_fuel

		dropped_oldest_frame = raw_window[0].copy()
		raw_window = np.concatenate([raw_window[1:], next_raw_frame[None, ...]], axis=0).astype(np.float32, copy=False)
		previous_frame_before_window = dropped_oldest_frame
		previous_true_surface_fuel = true_surface_fuel
		previous_true_canopy_fuel = true_canopy_fuel

	if mode_used_history:
		print(
			f"Rollout sample {split_sample_index} ({dataset_name}) completed with inference modes: "
			f"{', '.join(mode_used_history)}"
		)

	gif_path = None
	if bool(rollout_config["save_gif"]):
		sample_output_dir = output_root / split / dataset_name
		gif_path = sample_output_dir / (
			f"rollout_{dataset_name}_sample{int(split_sample_index):05d}_steps{len(step_records)}_{exogenous_mode}.gif"
		)
		_save_rollout_gif(
			step_records=step_records,
			output_path=gif_path,
			fps=fps,
			save_png_frames=save_png_frames,
		)
	return gif_path, metric_rows


def _select_sample_indices(
	dataset,
	num_samples: int,
	seed: int,
	start_indices: Sequence[int] | None,
) -> list[int]:
	"""Resolve rollout sample indices either explicitly or by reproducible random sampling."""

	if len(dataset) == 0:
		raise ValueError("Selected dataset split is empty; cannot run rollout.")

	if start_indices:
		resolved = [int(index) for index in start_indices]
		invalid = [index for index in resolved if index < 0 or index >= len(dataset)]
		if invalid:
			raise IndexError(
				f"start_indices contain invalid split sample indices. Valid range is [0, {len(dataset) - 1}], got {invalid}."
			)
		return resolved

	requested = max(1, int(num_samples))
	actual = min(requested, len(dataset))
	rng = np.random.default_rng(int(seed))
	return sorted(int(index) for index in rng.choice(len(dataset), size=actual, replace=False).tolist())


def rollout_predictions(
	config_path: str,
	split: str = "test",
	num_samples: int = 5,
	rollout_steps: int | None = None,
	seed: int | None = None,
	exogenous_mode: str | None = None,
	output_dir: str | None = None,
	fps: int | None = None,
	start_indices: Sequence[int] | None = None,
) -> list[Path]:
	"""Run multitask autoregressive rollouts and save one GIF per selected sample."""

	if torch is None:
		raise ImportError("PyTorch is required to run rollout predictions.")

	setup_logging()
	config = _ensure_config_path(load_config(config_path), config_path)
	if str(config.get("task_type", "regression")).lower() != "multitask":
		raise ValueError("rollout_predictions.py requires task_type='multitask'.")

	rollout_config = _resolve_rollout_config(config)
	selected_seed = int(seed if seed is not None else rollout_config["random_seed"])
	set_seed(selected_seed)

	if rollout_config["window_mode"] != "static":
		raise NotImplementedError(
			f"rollout.window_mode={rollout_config['window_mode']!r} is not implemented yet. Only 'static' is supported."
		)

	requested_rollout_steps = int(rollout_steps if rollout_steps is not None else rollout_config["rollout_steps"])
	if requested_rollout_steps <= 0:
		raise ValueError(f"rollout_steps must be positive, got {requested_rollout_steps}.")

	selected_exogenous_mode = str(exogenous_mode or rollout_config["exogenous_mode"]).lower()
	if selected_exogenous_mode not in {"teacher_forced", "constant"}:
		raise ValueError(
			f"rollout.exogenous_mode must be 'teacher_forced' or 'constant', got {selected_exogenous_mode!r}."
		)

	selected_fps = int(fps if fps is not None else rollout_config["fps"])
	dataset = _build_dataset_for_split(config, split)
	sampled_indices = _select_sample_indices(dataset, num_samples=num_samples, seed=selected_seed, start_indices=start_indices)

	device = _get_device(config)
	model = build_model_from_config(config, input_channels=int(getattr(dataset, "total_input_channels", 0)))
	checkpoint_path = _resolve_checkpoint_path(config)
	checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
	model.load_state_dict(checkpoint["model_state_dict"])
	model.to(device)
	model.eval()

	config_path_obj = Path(str(config.get("config_path", config.get("_config_path", config_path)))).expanduser().resolve()
	resolved_output_dir = _resolve_path(config_path_obj, output_dir or rollout_config["output_dir"])
	gif_paths: list[Path] = []
	metric_rows: list[dict[str, Any]] = []

	print(f"Selected split sample indices for rollout: {sampled_indices}")
	for split_sample_index in sampled_indices:
		gif_path, rows = _run_single_rollout(
			model=model,
			dataset=dataset,
			config=config,
			split=split,
			split_sample_index=split_sample_index,
			rollout_steps=requested_rollout_steps,
			exogenous_mode=selected_exogenous_mode,
			device=device,
			output_root=resolved_output_dir,
			fps=selected_fps,
			save_png_frames=bool(rollout_config["save_png_frames"]),
		)
		metric_rows.extend(rows)
		if gif_path is not None:
			gif_paths.append(gif_path)

	artifacts_dir = (config_path_obj.parent / "artifacts" / "logs").resolve()
	metrics_csv_path = artifacts_dir / f"rollout_metrics_{str(split).lower()}.csv"
	_write_metrics_csv(metrics_csv_path, metric_rows)
	print(f"Saved rollout metrics to {metrics_csv_path}")
	for gif_path in gif_paths:
		print(f"Saved GIF: {gif_path}")
	return gif_paths


def build_arg_parser() -> argparse.ArgumentParser:
	"""Create the CLI parser."""

	parser = argparse.ArgumentParser(description="Generate multitask autoregressive rollout GIFs.")
	parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the experiment config.")
	parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"), help="Dataset split to roll out.")
	parser.add_argument("--num_samples", type=int, default=5, help="Number of random split samples to roll out.")
	parser.add_argument("--rollout_steps", type=int, default=None, help="Optional override for rollout.rollout_steps.")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sample selection.")
	parser.add_argument(
		"--exogenous_mode",
		type=str,
		default=None,
		choices=("teacher_forced", "constant"),
		help="Optional override for rollout.exogenous_mode.",
	)
	parser.add_argument("--output_dir", type=str, default=None, help="Optional override for rollout.output_dir.")
	parser.add_argument("--fps", type=int, default=None, help="Frames per second for the saved GIFs.")
	parser.add_argument(
		"--start_indices",
		type=int,
		nargs="+",
		default=None,
		help="Explicit split sample indices to roll out. If provided, random selection is ignored.",
	)
	return parser


def main() -> None:
	"""CLI entry point."""

	args = build_arg_parser().parse_args()
	rollout_predictions(
		config_path=args.config,
		split=args.split,
		num_samples=args.num_samples,
		rollout_steps=args.rollout_steps,
		seed=args.seed,
		exogenous_mode=args.exogenous_mode,
		output_dir=args.output_dir,
		fps=args.fps,
		start_indices=args.start_indices,
	)


if __name__ == "__main__":
	main()
