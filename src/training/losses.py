"""Loss functions for wildfire forecasting.

Supports two initial task modes:
- intensity regression
- binary perimeter / fire mask segmentation
"""

from __future__ import annotations

from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
	import torch.nn.functional as F  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)
	F = None


def _get_section(config, *names):
	"""Return the first mapping-like section present in ``config``."""

	if isinstance(config, dict):
		for name in names:
			section = config.get(name)
			if isinstance(section, dict):
				return section
	return config if isinstance(config, dict) else {}


def _sigmoid_if_needed(y_pred: torch.Tensor, from_logits: bool) -> torch.Tensor:
	"""Convert logits to probabilities when needed."""

	return torch.sigmoid(y_pred) if from_logits else y_pred


class WeightedMSELoss(nn.Module):
	"""Weighted mean squared error for regression.

	Pixels with ``target > active_threshold`` receive a higher weight.
	"""

	def __init__(self, active_threshold: float = 0.0, active_weight: float = 2.0) -> None:
		super().__init__()
		self.active_threshold = float(active_threshold)
		self.active_weight = float(active_weight)

	def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
		if y_pred.shape != y_true.shape:
			raise ValueError(f"WeightedMSELoss expects matching shapes, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}.")

		base_error = (y_pred - y_true) ** 2
		active_mask = (y_true > self.active_threshold).to(dtype=y_pred.dtype)
		weights = 1.0 + active_mask * (self.active_weight - 1.0)
		return (base_error * weights).mean()


class DiceLoss(nn.Module):
	"""Soft Dice loss for binary segmentation.

	``y_pred`` may contain logits or probabilities depending on ``from_logits``.
	"""

	def __init__(self, from_logits: bool = True, eps: float = 1e-6) -> None:
		super().__init__()
		self.from_logits = bool(from_logits)
		self.eps = float(eps)

	def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
		if y_pred.shape != y_true.shape:
			raise ValueError(f"DiceLoss expects matching shapes, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}.")

		probabilities = _sigmoid_if_needed(y_pred, self.from_logits)
		probabilities = probabilities.reshape(probabilities.shape[0], -1)
		targets = y_true.reshape(y_true.shape[0], -1)

		intersection = (probabilities * targets).sum(dim=1)
		denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
		dice_score = (2.0 * intersection + self.eps) / (denominator + self.eps)
		return 1.0 - dice_score.mean()


class BCEDiceLoss(nn.Module):
	"""Combination of BCE and Dice loss for binary segmentation."""

	def __init__(self, from_logits: bool = True, bce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-6) -> None:
		super().__init__()
		self.from_logits = bool(from_logits)
		self.bce_weight = float(bce_weight)
		self.dice_weight = float(dice_weight)
		self.eps = float(eps)

	def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
		if y_pred.shape != y_true.shape:
			raise ValueError(
				f"BCEDiceLoss expects matching shapes, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}."
			)

		if self.from_logits:
			bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
		else:
			bce_loss = F.binary_cross_entropy(y_pred, y_true)

		dice_loss = DiceLoss(from_logits=self.from_logits, eps=self.eps)(y_pred, y_true)
		return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
	"""Binary focal loss.

	Useful for class imbalance in perimeter segmentation. Supports logits or probabilities.
	"""

	def __init__(self, from_logits: bool = True, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-6) -> None:
		super().__init__()
		self.from_logits = bool(from_logits)
		self.alpha = float(alpha)
		self.gamma = float(gamma)
		self.eps = float(eps)

	def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
		if y_pred.shape != y_true.shape:
			raise ValueError(f"FocalLoss expects matching shapes, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}.")

		if self.from_logits:
			bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
			probabilities = torch.sigmoid(y_pred)
		else:
			probabilities = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
			bce = F.binary_cross_entropy(probabilities, y_true, reduction="none")

		pt = torch.where(y_true > 0.5, probabilities, 1.0 - probabilities)
		alpha_factor = torch.where(y_true > 0.5, self.alpha, 1.0 - self.alpha)
		focal_factor = (1.0 - pt).pow(self.gamma)
		return (alpha_factor * focal_factor * bce).mean()


def get_loss_function(config):
	"""Build the configured loss function.

	Supported ``loss_type`` values:
	- regression: ``mse``, ``mae``, ``huber``, ``weighted_mse``
	- segmentation: ``bce_with_logits``, ``dice``, ``bce_dice``, ``focal``
	"""

	if torch is None or F is None:
		raise ImportError("PyTorch is required to build wildfire loss functions.")

	model_config = _get_section(config, "model")
	training_config = _get_section(config, "training")
	loss_config = _get_section(config, "loss")

	task_type = str(
		loss_config.get(
			"task_type",
			training_config.get("task_type", model_config.get("task_type", config.get("task_type", "regression"))),
		)
	).lower()
	loss_type = str(
		loss_config.get(
			"loss_type",
			training_config.get(
				"loss_type",
				model_config.get(
					"loss_type",
					config.get("loss_type", "bce_dice" if task_type == "segmentation" else "huber"),
				),
			),
		)
	).lower()

	if task_type == "regression":
		if loss_type == "mse":
			return nn.MSELoss()
		if loss_type == "mae":
			return nn.L1Loss()
		if loss_type == "huber":
			delta = float(loss_config.get("huber_delta", training_config.get("huber_delta", 1.0)))
			return nn.SmoothL1Loss(beta=delta)
		if loss_type == "weighted_mse":
			active_threshold = float(loss_config.get("active_threshold", training_config.get("active_threshold", 0.0)))
			active_weight = float(loss_config.get("active_weight", training_config.get("active_weight", 2.0)))
			return WeightedMSELoss(active_threshold=active_threshold, active_weight=active_weight)
		raise ValueError(f"Unsupported regression loss_type: {loss_type}")

	if task_type == "segmentation":
		from_logits = bool(loss_config.get("from_logits", training_config.get("from_logits", True)))
		if loss_type == "bce_with_logits":
			return nn.BCEWithLogitsLoss()
		if loss_type == "dice":
			return DiceLoss(from_logits=from_logits)
		if loss_type in {"bce_dice", "bce+dice", "bce_dice_loss"}:
			bce_weight = float(loss_config.get("bce_weight", training_config.get("bce_weight", 1.0)))
			dice_weight = float(loss_config.get("dice_weight", training_config.get("dice_weight", 1.0)))
			return BCEDiceLoss(from_logits=from_logits, bce_weight=bce_weight, dice_weight=dice_weight)
		if loss_type == "focal":
			alpha = float(loss_config.get("alpha", training_config.get("alpha", 0.25)))
			gamma = float(loss_config.get("gamma", training_config.get("gamma", 2.0)))
			return FocalLoss(from_logits=from_logits, alpha=alpha, gamma=gamma)
		raise ValueError(f"Unsupported segmentation loss_type: {loss_type}")

	raise ValueError(f"Unsupported task_type: {task_type}")
