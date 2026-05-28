"""Metrics for wildfire forecasting."""

from __future__ import annotations

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None


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
			if y_pred.ndim != 4 or y_pred.shape[1] != 3:
				raise ValueError(f"Multitask metrics expect tensors shaped (B, 3, H, W), got {tuple(y_pred.shape)}.")

			pred_surface = y_pred[:, 0:1]
			true_surface = y_true[:, 0:1]
			pred_canopy = y_pred[:, 1:2]
			true_canopy = y_true[:, 1:2]
			mask_logits = y_pred[:, 2:3]
			true_mask = y_true[:, 2:3].to(dtype=torch.float32)
			mask_prob = torch.sigmoid(mask_logits)
			mask_pred = (mask_prob > 0.5).to(dtype=torch.float32)

			surface_abs_error = torch.abs(pred_surface - true_surface)
			canopy_abs_error = torch.abs(pred_canopy - true_canopy)
			active_mask = true_mask > 0.5

			if active_mask.any():
				active_surface_mae = surface_abs_error[active_mask].mean()
				active_canopy_mae = canopy_abs_error[active_mask].mean()
			else:
				active_surface_mae = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
				active_canopy_mae = torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)

			segmentation_metrics = _segmentation_stats(mask_pred, true_mask, eps)
			return {
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
			}

		raise ValueError(f"Unsupported task_type: {task_type}")
