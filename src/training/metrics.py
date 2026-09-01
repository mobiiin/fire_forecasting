"""Metrics for wildfire forecasting."""

from __future__ import annotations

import math

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None

from src.data.energy_release import resolve_energy_output_channel_names, resolve_energy_release_config
from src.training.model_outputs import extract_prediction


def _get_section(config, *names):
	"""Return the first mapping-like section present in ``config``."""

	if isinstance(config, dict):
		for name in names:
			section = config.get(name)
			if isinstance(section, dict):
				return section
	return config if isinstance(config, dict) else {}


def _as_probabilities(y_pred: torch.Tensor, from_logits: bool) -> torch.Tensor:
	"""Convert logits to probabilities when needed."""

	return torch.sigmoid(y_pred) if from_logits else y_pred


def _segmentation_stats(predicted_mask: torch.Tensor, target_mask: torch.Tensor, eps: float) -> dict[str, float]:
	"""Compute common binary-mask metrics from float masks."""

	true_positive = torch.sum(predicted_mask * target_mask)
	false_positive = torch.sum(predicted_mask * (1.0 - target_mask))
	false_negative = torch.sum((1.0 - predicted_mask) * target_mask)
	accuracy = torch.mean((predicted_mask == target_mask).to(dtype=torch.float32))
	iou = true_positive / (true_positive + false_positive + false_negative + eps)
	dice = (2.0 * true_positive) / (2.0 * true_positive + false_positive + false_negative + eps)
	precision = true_positive / (true_positive + false_positive + eps)
	recall = true_positive / (true_positive + false_negative + eps)
	return {
		"accuracy": float(accuracy.item()),
		"iou": float(iou.item()),
		"dice": float(dice.item()),
		"precision": float(precision.item()),
		"recall": float(recall.item()),
	}


def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, config) -> dict[str, float]:
	"""Compute task-specific metrics and return them as Python floats."""

	if torch is None:
		raise ImportError("PyTorch is required to compute wildfire metrics.")
	y_pred = extract_prediction(y_pred)
	if y_pred.shape != y_true.shape:
		raise ValueError(f"Metrics expect matching shapes, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}.")

	model_config = _get_section(config, "model")
	training_config = _get_section(config, "training")
	metric_config = _get_section(config, "metrics")
	task_type = str(
		metric_config.get(
			"task_type",
			training_config.get("task_type", model_config.get("task_type", config.get("task_type", "regression"))),
		)
	).lower()
	eps = float(metric_config.get("eps", training_config.get("eps", 1e-6)))

	with torch.no_grad():
		if task_type == "regression":
			active_threshold = float(
				metric_config.get(
					"active_threshold",
					training_config.get("active_threshold", config.get("active_threshold", config.get("fire_threshold", 0.0))),
				)
			)
			abs_error = torch.abs(y_pred - y_true)
			mae = abs_error.mean()
			rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2) + eps)
			active_mask = y_true > active_threshold
			if active_mask.any():
				active_region_mae = abs_error[active_mask].mean()
			else:
				active_region_mae = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
			return {
				"mae": float(mae.item()),
				"rmse": float(rmse.item()),
				"active_mae": float(active_region_mae.item()),
				"active_region_mae": float(active_region_mae.item()),
			}

		if task_type == "segmentation":
			from_logits = bool(metric_config.get("from_logits", training_config.get("from_logits", True)))
			prediction_threshold = float(metric_config.get("prediction_threshold", training_config.get("prediction_threshold", 0.5)))
			target_threshold = float(metric_config.get("target_threshold", training_config.get("target_threshold", config.get("fire_threshold", 0.5))))
			probabilities = _as_probabilities(y_pred, from_logits)
			predicted_mask = (probabilities >= prediction_threshold).to(dtype=torch.float32)
			target_mask = (y_true >= target_threshold).to(dtype=torch.float32)
			return _segmentation_stats(predicted_mask, target_mask, eps)

		if task_type == "multitask":
			processed_mode = str(_get_section(config, "dataloader").get("source", "")).lower() == "processed_full_frames"
			energy_release = resolve_energy_release_config(config)
			energy_output_names = resolve_energy_output_channel_names(config)
			expected_channels = 4 if processed_mode else 3 + len(energy_output_names)
			if y_pred.ndim != 4 or y_pred.shape[1] != expected_channels:
				raise ValueError(
					f"Multitask metrics expect tensors shaped (B, {expected_channels}, H, W), got {tuple(y_pred.shape)}."
				)

			pred_surface = y_pred[:, 0:1]
			true_surface = y_true[:, 0:1]
			pred_canopy = y_pred[:, 1:2]
			true_canopy = y_true[:, 1:2]
			mask_logits = y_pred[:, 2:3]
			true_mask = y_true[:, 2:3].to(dtype=torch.float32)
			if not torch.isfinite(true_mask).all() or bool(((true_mask < 0) | (true_mask > 1)).any()):
				raise ValueError("Fire-mask targets must be finite floats in [0, 1].")
			mask_prob = torch.sigmoid(mask_logits)
			mask_pred = (mask_prob > 0.5).to(dtype=torch.float32)

			surface_abs_error = torch.abs(pred_surface - true_surface)
			canopy_abs_error = torch.abs(pred_canopy - true_canopy)
			active_mask = true_mask > 0.5

			if active_mask.any():
				active_surface_mae = surface_abs_error[active_mask].mean()
				active_canopy_mae = canopy_abs_error[active_mask].mean()
			else:
				active_surface_mae = torch.full((), float("nan"), device=y_pred.device, dtype=y_pred.dtype)
				active_canopy_mae = torch.full((), float("nan"), device=y_pred.device, dtype=y_pred.dtype)

			segmentation_metrics = _segmentation_stats(mask_pred, true_mask, eps)
			results = {
				"surface_consumed_mae": float(surface_abs_error.mean().item()),
				"surface_consumed_rmse": float(torch.sqrt(torch.mean((pred_surface - true_surface) ** 2) + eps).item()),
				"active_surface_consumed_mae": float(active_surface_mae.item()),
				"canopy_consumed_mae": float(canopy_abs_error.mean().item()),
				"canopy_consumed_rmse": float(torch.sqrt(torch.mean((pred_canopy - true_canopy) ** 2) + eps).item()),
				"active_canopy_consumed_mae": float(active_canopy_mae.item()),
				"mask_iou": float(segmentation_metrics["iou"]),
				"mask_dice": float(segmentation_metrics["dice"]),
				"mask_precision": float(segmentation_metrics["precision"]),
				"mask_recall": float(segmentation_metrics["recall"]),
				"active_mask_fraction": float(true_mask.mean().item()),
				"predicted_active_mask_fraction": float(mask_pred.mean().item()),
				"surface_mae": float(surface_abs_error.mean().item()),
				"surface_rmse": float(torch.sqrt(torch.mean((pred_surface - true_surface) ** 2) + eps).item()),
				"canopy_mae": float(canopy_abs_error.mean().item()),
				"canopy_rmse": float(torch.sqrt(torch.mean((pred_canopy - true_canopy) ** 2) + eps).item()),
			}
			# Patch-level no-fire metrics expose false positives that aggregate Dice can hide.
			active_fraction = true_mask.flatten(1).mean(dim=1)
			no_fire_patches = active_fraction < float(metric_config.get("no_fire_active_fraction_threshold", 0.001))
			active_patches = ~no_fire_patches
			results["no_fire_patch_count"] = float(no_fire_patches.sum().item())
			results["active_patch_count"] = float(active_patches.sum().item())
			if no_fire_patches.any():
				results["no_fire_mask_prob_mean"] = float(mask_prob[no_fire_patches].mean().item())
				results["no_fire_mask_false_positive_rate"] = float(mask_pred[no_fire_patches].mean().item())
				results["no_fire_surface_pred_mean"] = float(pred_surface[no_fire_patches].mean().item())
				results["no_fire_canopy_pred_mean"] = float(pred_canopy[no_fire_patches].mean().item())
			else:
				for name in ("no_fire_mask_prob_mean", "no_fire_mask_false_positive_rate", "no_fire_surface_pred_mean", "no_fire_canopy_pred_mean"): results[name] = math.nan
			if energy_output_names:
				pred_energy_log = y_pred[:, 3:4]
				true_energy_log = y_true[:, 3:4]
				energy_log_error = pred_energy_log - true_energy_log
				energy_log_abs_error = torch.abs(energy_log_error)
				results["energy_log_mae"] = float(torch.mean(torch.abs(energy_log_error)).item())
				results["energy_log_rmse"] = float(torch.sqrt(torch.mean(energy_log_error ** 2) + eps).item())

				if str(energy_release.get("target_transform", "log1p")) == "log1p":
					pred_energy_MW = torch.expm1(pred_energy_log)
					true_energy_MW = torch.expm1(true_energy_log)
				else:
					pred_energy_MW = pred_energy_log
					true_energy_MW = true_energy_log
				pred_energy_MW = torch.clamp(pred_energy_MW, min=0.0)
				true_energy_MW = torch.clamp(true_energy_MW, min=0.0)

				energy_abs_error = torch.abs(pred_energy_MW - true_energy_MW)
				results["energy_MW_mae"] = float(torch.mean(energy_abs_error).item())
				results["energy_MW_rmse"] = float(torch.sqrt(torch.mean((pred_energy_MW - true_energy_MW) ** 2) + eps).item())
				results["energy_mw_mae"] = results["energy_MW_mae"]
				results["energy_mw_rmse"] = results["energy_MW_rmse"]

				energy_active_threshold_MW = float(_get_section(config, "multitask").get("energy_active_threshold_MW", 0.001))
				consumed_active_threshold = float(
					_get_section(config, "multitask").get(
						"consumed_active_threshold",
						_get_section(config, "multitask").get("consumed_fuel_threshold", 0.001),
					)
				)
				energy_active_threshold_log = math.log1p(max(energy_active_threshold_MW, 0.0))
				if processed_mode:
					target_defined_active = true_mask > 0.5
				else:
					target_defined_active = (
						(true_mask > 0.5)
						| (true_energy_log > energy_active_threshold_log)
						| (true_surface > consumed_active_threshold)
						| (true_canopy > consumed_active_threshold)
					)
				if target_defined_active.any():
					results["active_energy_log_mae"] = float(energy_log_abs_error[target_defined_active].mean().item())
					results["active_energy_log_rmse"] = float(
						torch.sqrt(torch.mean(energy_log_error[target_defined_active] ** 2) + eps).item()
					)
				else:
					results["active_energy_log_mae"] = math.nan
					results["active_energy_log_rmse"] = math.nan
				true_energy_active = (true_energy_MW > energy_active_threshold_MW).to(dtype=torch.float32)
				pred_energy_active = (pred_energy_MW > energy_active_threshold_MW).to(dtype=torch.float32)
				if true_energy_active.any():
					results["energy_MW_active_mae"] = float(energy_abs_error[true_energy_active > 0.5].mean().item())
					results["energy_MW_active_rmse"] = float(
						torch.sqrt(torch.mean((pred_energy_MW[true_energy_active > 0.5] - true_energy_MW[true_energy_active > 0.5]) ** 2) + eps).item()
					)
				else:
					results["energy_MW_active_mae"] = 0.0
					results["energy_MW_active_rmse"] = 0.0
				results["active_energy_mw_mae"] = results["energy_MW_active_mae"]
				results["active_energy_mw_rmse"] = results["energy_MW_active_rmse"]

				energy_active_metrics = _segmentation_stats(pred_energy_active, true_energy_active, eps)
				results["energy_active_iou"] = float(energy_active_metrics["iou"])
				results["energy_active_dice"] = float(energy_active_metrics["dice"])
				results["energy_active_precision"] = float(energy_active_metrics["precision"])
				results["energy_active_recall"] = float(energy_active_metrics["recall"])
				true_sum = torch.sum(true_energy_MW)
				pred_sum = torch.sum(pred_energy_MW)
				results["energy_total_true_MW_sum"] = float(true_sum.item())
				results["energy_total_pred_MW_sum"] = float(pred_sum.item())
				results["energy_total_sum_relative_error"] = float((torch.abs(pred_sum - true_sum) / (torch.abs(true_sum) + eps)).item())
				results["energy_active_fraction"] = float(true_energy_active.mean().item())
				if no_fire_patches.any(): results["no_fire_energy_log_pred_mean"] = float(pred_energy_log[no_fire_patches].mean().item())
				else: results["no_fire_energy_log_pred_mean"] = math.nan
			return results

		raise ValueError(f"Unsupported task_type: {task_type}")
