"""Loss functions for wildfire forecasting."""

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


def _weighted_mean(loss_map: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
	"""Compute a weighted mean with safe normalization."""

	if loss_map.shape != weights.shape:
		raise ValueError(f"loss_map and weights must match, got {tuple(loss_map.shape)} vs {tuple(weights.shape)}.")
	return torch.sum(loss_map * weights) / torch.clamp(weights.sum(), min=1.0)


def _build_weight_map(active_mask: torch.Tensor, active_weight: float, background_weight: float) -> torch.Tensor:
	"""Create a floating-point weight map from a binary active mask."""

	return torch.where(
		active_mask,
		torch.full_like(active_mask, float(active_weight), dtype=torch.float32),
		torch.full_like(active_mask, float(background_weight), dtype=torch.float32),
	).to(dtype=torch.float32)


class WeightedMSELoss(nn.Module):
	"""Weighted mean squared error for regression."""

	def __init__(self, active_threshold: float = 0.0, active_weight: float = 2.0, background_weight: float = 1.0) -> None:
		super().__init__()
		self.active_threshold = float(active_threshold)
		self.active_weight = float(active_weight)
		self.background_weight = float(background_weight)

	def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
		if y_pred.shape != y_true.shape:
			raise ValueError(f"WeightedMSELoss expects matching shapes, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}.")
		active_mask = y_true > self.active_threshold
		weights = _build_weight_map(active_mask, self.active_weight, self.background_weight).to(device=y_pred.device, dtype=y_pred.dtype)
		return _weighted_mean((y_pred - y_true) ** 2, weights)


class WeightedHuberLoss(nn.Module):
	"""Weighted SmoothL1 / Huber loss for regression."""

	def __init__(
		self,
		active_threshold: float = 0.0,
		active_weight: float = 2.0,
		background_weight: float = 1.0,
		delta: float = 1.0,
	) -> None:
		super().__init__()
		self.active_threshold = float(active_threshold)
		self.active_weight = float(active_weight)
		self.background_weight = float(background_weight)
		self.delta = float(delta)

	def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
		if y_pred.shape != y_true.shape:
			raise ValueError(f"WeightedHuberLoss expects matching shapes, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}.")
		active_mask = y_true > self.active_threshold
		weights = _build_weight_map(active_mask, self.active_weight, self.background_weight).to(device=y_pred.device, dtype=y_pred.dtype)
		loss_map = F.smooth_l1_loss(y_pred, y_true, reduction="none", beta=self.delta)
		return _weighted_mean(loss_map, weights)


class DiceLoss(nn.Module):
	"""Soft Dice loss for binary segmentation."""

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
			raise ValueError(f"BCEDiceLoss expects matching shapes, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}.")
		if self.from_logits:
			bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
		else:
			bce_loss = F.binary_cross_entropy(y_pred, y_true)
		dice_loss = DiceLoss(from_logits=self.from_logits, eps=self.eps)(y_pred, y_true)
		return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
	"""Binary focal loss."""

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


class MultiTaskLoss(nn.Module):
	"""Loss for multitask surface/canopy consumed fuel + active mask prediction."""

	def __init__(self, config) -> None:
		super().__init__()
		self.config = config
		self.multitask_config = _get_section(config, "multitask")
		self.training_config = _get_section(config, "training")
		self.loss_config = _get_section(config, "loss")
		self.segmentation_loss_name = str(self.multitask_config.get("segmentation_loss", "bce_dice")).lower()
		self.regression_loss_name = str(self.multitask_config.get("regression_loss", "weighted_huber")).lower()
		self.surface_loss_weight = float(self.multitask_config.get("surface_loss_weight", 1.0))
		self.canopy_loss_weight = float(self.multitask_config.get("canopy_loss_weight", 1.0))
		self.segmentation_loss_weight = float(self.multitask_config.get("segmentation_loss_weight", 1.0))
		self.active_fire_weight = float(self.multitask_config.get("active_fire_weight", 10.0))
		self.background_weight = float(self.multitask_config.get("background_weight", 1.0))
		self.consumed_fuel_threshold = float(self.multitask_config.get("consumed_fuel_threshold", 0.01))
		self.huber_delta = float(self.multitask_config.get("huber_delta", self.training_config.get("huber_delta", 1.0)))
		self.dice_eps = float(self.multitask_config.get("dice_eps", self.training_config.get("eps", 1e-6)))

	def _regression_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
		"""Compute one multitask regression-channel loss."""

		if y_pred.shape != y_true.shape or y_pred.shape != true_mask.shape:
			raise ValueError(
				f"Multitask regression loss expects matching shapes, got pred={tuple(y_pred.shape)} "
				f"true={tuple(y_true.shape)} mask={tuple(true_mask.shape)}."
			)

		active_mask = (true_mask > 0.5) | (y_true > self.consumed_fuel_threshold)
		weights = _build_weight_map(active_mask, self.active_fire_weight, self.background_weight).to(device=y_pred.device, dtype=y_pred.dtype)
		loss_name = self.regression_loss_name
		if loss_name == "mse":
			return F.mse_loss(y_pred, y_true)
		if loss_name == "huber":
			return F.smooth_l1_loss(y_pred, y_true, beta=self.huber_delta)
		if loss_name == "weighted_mse":
			return _weighted_mean((y_pred - y_true) ** 2, weights)
		if loss_name == "weighted_huber":
			return _weighted_mean(F.smooth_l1_loss(y_pred, y_true, reduction="none", beta=self.huber_delta), weights)
		raise ValueError(
			"Unsupported multitask regression_loss. "
			f"Expected one of 'weighted_huber', 'weighted_mse', 'huber', 'mse', got {loss_name!r}."
		)

	def _mask_loss(self, pred_logits: torch.Tensor, true_mask: torch.Tensor) -> torch.Tensor:
		"""Compute the multitask segmentation-channel loss."""

		if pred_logits.shape != true_mask.shape:
			raise ValueError(f"Mask loss expects matching shapes, got {tuple(pred_logits.shape)} and {tuple(true_mask.shape)}.")

		weights = _build_weight_map(true_mask > 0.5, self.active_fire_weight, self.background_weight).to(device=pred_logits.device, dtype=pred_logits.dtype)
		if self.segmentation_loss_name == "bce_dice":
			bce = _weighted_mean(F.binary_cross_entropy_with_logits(pred_logits, true_mask, reduction="none"), weights)
			dice = DiceLoss(from_logits=True, eps=self.dice_eps)(pred_logits, true_mask)
			return bce + dice
		if self.segmentation_loss_name == "bce_with_logits":
			return _weighted_mean(F.binary_cross_entropy_with_logits(pred_logits, true_mask, reduction="none"), weights)
		if self.segmentation_loss_name == "dice":
			return DiceLoss(from_logits=True, eps=self.dice_eps)(pred_logits, true_mask)
		if self.segmentation_loss_name == "focal":
			return FocalLoss(from_logits=True)(pred_logits, true_mask)
		raise ValueError(
			"Unsupported multitask segmentation_loss. "
			f"Expected one of 'bce_dice', 'bce_with_logits', 'dice', 'focal', got {self.segmentation_loss_name!r}."
		)

	def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict[str, torch.Tensor]:
		if y_pred.shape != y_true.shape:
			raise ValueError(f"MultiTaskLoss expects matching shapes, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}.")
		if y_pred.ndim != 4 or y_pred.shape[1] != 3:
			raise ValueError(f"MultiTaskLoss expects tensors shaped (B, 3, H, W), got {tuple(y_pred.shape)}.")

		pred_surface_consumed = y_pred[:, 0:1]
		true_surface_consumed = y_true[:, 0:1]
		pred_canopy_consumed = y_pred[:, 1:2]
		true_canopy_consumed = y_true[:, 1:2]
		pred_mask_logits = y_pred[:, 2:3]
		true_mask = y_true[:, 2:3]

		surface_loss = self._regression_loss(pred_surface_consumed, true_surface_consumed, true_mask)
		canopy_loss = self._regression_loss(pred_canopy_consumed, true_canopy_consumed, true_mask)
		mask_loss = self._mask_loss(pred_mask_logits, true_mask)
		total_loss = (
			self.surface_loss_weight * surface_loss
			+ self.canopy_loss_weight * canopy_loss
			+ self.segmentation_loss_weight * mask_loss
		)
		return {
			"total_loss": total_loss,
			"surface_loss": surface_loss,
			"canopy_loss": canopy_loss,
			"mask_loss": mask_loss,
		}


def get_loss_function(config):
	"""Build the configured loss function."""

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
				model_config.get("loss_type", config.get("loss_type", "bce_dice" if task_type == "segmentation" else "huber")),
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
			background_weight = float(loss_config.get("background_weight", training_config.get("background_weight", 1.0)))
			return WeightedMSELoss(active_threshold=active_threshold, active_weight=active_weight, background_weight=background_weight)
		if loss_type == "weighted_huber":
			active_threshold = float(loss_config.get("active_threshold", training_config.get("active_threshold", 0.0)))
			active_weight = float(loss_config.get("active_weight", training_config.get("active_weight", 2.0)))
			background_weight = float(loss_config.get("background_weight", training_config.get("background_weight", 1.0)))
			delta = float(loss_config.get("huber_delta", training_config.get("huber_delta", 1.0)))
			return WeightedHuberLoss(
				active_threshold=active_threshold,
				active_weight=active_weight,
				background_weight=background_weight,
				delta=delta,
			)
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

	if task_type == "multitask":
		return MultiTaskLoss(config)

	raise ValueError(f"Unsupported task_type: {task_type}")
