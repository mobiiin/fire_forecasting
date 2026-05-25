"""Metrics for wildfire forecasting."""

from __future__ import annotations

from types import SimpleNamespace

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


def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, config) -> dict[str, float]:
	"""Compute task-specific metrics and return them as Python floats.

	Regression metrics:
	- MAE
	- RMSE
	- active_region_mae

	Segmentation metrics:
	- IoU
	- Dice
	- precision
	- recall
	"""

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
					training_config.get(
						"active_threshold",
						config.get("active_threshold", config.get("fire_threshold", 0.0)),
					),
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
			prediction_threshold = float(
				metric_config.get(
					"prediction_threshold",
					training_config.get("prediction_threshold", 0.5),
				)
			)
			target_threshold = float(
				metric_config.get(
					"target_threshold",
					training_config.get("target_threshold", config.get("fire_threshold", 0.5)),
				)
			)

			probabilities = _as_probabilities(y_pred, from_logits)
			predicted_mask = probabilities >= prediction_threshold
			target_mask = y_true >= target_threshold

			predicted_mask = predicted_mask.to(dtype=torch.float32)
			target_mask = target_mask.to(dtype=torch.float32)

			true_positive = torch.sum(predicted_mask * target_mask)
			false_positive = torch.sum(predicted_mask * (1.0 - target_mask))
			false_negative = torch.sum((1.0 - predicted_mask) * target_mask)

			iou = true_positive / (true_positive + false_positive + false_negative + eps)
			dice = (2.0 * true_positive) / (2.0 * true_positive + false_positive + false_negative + eps)
			precision = true_positive / (true_positive + false_positive + eps)
			recall = true_positive / (true_positive + false_negative + eps)

			return {
				"iou": float(iou.item()),
				"dice": float(dice.item()),
				"precision": float(precision.item()),
				"recall": float(recall.item()),
			}

		raise ValueError(f"Unsupported task_type: {task_type}")
