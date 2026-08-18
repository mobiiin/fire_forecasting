"""Loss functions for wildfire forecasting."""

from __future__ import annotations

import math
from types import SimpleNamespace

try:
	import torch  # type: ignore[import-not-found]
	import torch.nn as nn  # type: ignore[import-not-found]
	import torch.nn.functional as F  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	nn = SimpleNamespace(Module=object)
	F = None

from src.data.energy_release import resolve_energy_output_channel_names, resolve_energy_release_config



from src.training.model_outputs import extract_aux_outputs, extract_prediction

def _get_section(config, *names):
	"""Return the first mapping-like section present in ``config``."""

	if isinstance(config, dict):
		for name in names:
			section = config.get(name)
			if isinstance(section, dict):
				return section
	return config if isinstance(config, dict) else {}


def _get_training_loss_section(config) -> dict:
	"""Return nested training.loss config when present."""

	if isinstance(config, dict):
		training = config.get("training")
		if isinstance(training, dict):
			loss = training.get("loss")
			if isinstance(loss, dict):
				return loss
	return {}


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
		self.training_loss_config = _get_training_loss_section(config)
		self.model_config = _get_section(config, "model")
		self.architecture = str(self.model_config.get("architecture", self.model_config.get("name", ""))).lower()
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
		self.energy_release_config = resolve_energy_release_config(config)
		self.energy_output_names = resolve_energy_output_channel_names(config)
		self.expected_channels = 3 + len(self.energy_output_names)
		self.energy_loss_weight = float(self.multitask_config.get("energy_loss_weight", 1.0))
		self.energy_loss_name = str(self.multitask_config.get("energy_loss", "huber")).lower()
		self.energy_loss_space = str(self.multitask_config.get("energy_loss_space", "log")).lower()
		self.energy_active_weight = float(self.multitask_config.get("energy_active_weight", 10.0))
		self.energy_background_weight = float(self.multitask_config.get("energy_background_weight", 1.0))
		self.processed_full_frames = str(_get_section(config, "dataloader").get("source", "")).lower() == "processed_full_frames"
		if self.processed_full_frames:
			self.regression_loss_name = "huber"
			self.surface_loss_weight = 1.0
			self.canopy_loss_weight = 1.0
			self.segmentation_loss_weight = 5.0
			self.energy_loss_weight = 1.0
			self.active_fire_weight = 1.0
			self.background_weight = 1.0
			self.energy_active_weight = 1.0
			self.energy_background_weight = 1.0
		self.energy_active_threshold_MW = float(self.multitask_config.get("energy_active_threshold_MW", 0.001))
		background_suppression_config = {}
		if isinstance(self.loss_config.get("background_suppression"), dict):
			background_suppression_config.update(self.loss_config.get("background_suppression", {}))
		if isinstance(self.training_loss_config.get("background_suppression"), dict):
			background_suppression_config.update(self.training_loss_config.get("background_suppression", {}))
		self.background_suppression_config = background_suppression_config
		self.background_suppression_enabled = bool(background_suppression_config.get("enabled", False))
		background_architectures = background_suppression_config.get("architectures")
		if background_architectures is not None:
			allowed_architectures = {str(name).lower() for name in background_architectures}
			self.background_suppression_enabled = self.background_suppression_enabled and self.architecture in allowed_architectures
		self.background_suppression_weight = float(background_suppression_config.get("weight", 0.0))
		self.background_suppression_include_surface = bool(background_suppression_config.get("include_surface", True))
		self.background_suppression_include_canopy = bool(background_suppression_config.get("include_canopy", True))
		self.background_suppression_include_energy = bool(background_suppression_config.get("include_energy", True))
		self.background_suppression_include_mask_prob = bool(background_suppression_config.get("include_mask_prob", True))
		self.background_suppression_inactive_definition = str(background_suppression_config.get("inactive_definition", "combined")).lower()
		if self.background_suppression_inactive_definition not in ("combined", "mask_only"):
			raise ValueError(
				"training.loss.background_suppression.inactive_definition must be 'combined' or 'mask_only', "
				f"got {self.background_suppression_inactive_definition!r}."
			)
		self.background_suppression_consumed_threshold = float(background_suppression_config.get("consumed_threshold", 0.001))
		self.background_suppression_energy_log_threshold = float(background_suppression_config.get("energy_log_threshold", 0.001))
		self.background_suppression_mask_threshold = float(background_suppression_config.get("mask_threshold", 0.5))
		self.background_suppression_reduction = str(background_suppression_config.get("reduction", "mean")).lower()
		if self.background_suppression_reduction != "mean":
			raise ValueError("training.loss.background_suppression.reduction currently supports only 'mean'.")
		self.cawfe_latte_loss_enabled = self.architecture == "cawfe_latte"
		if self.cawfe_latte_loss_enabled:
			surface_config = _get_section(self.training_loss_config, "surface")
			canopy_config = _get_section(self.training_loss_config, "canopy")
			mask_config = _get_section(self.training_loss_config, "mask")
			energy_config = _get_section(self.training_loss_config, "energy")
			aux_config = _get_section(self.training_loss_config, "auxiliary_fire_support")
			self.surface_loss_weight = float(surface_config.get("weight", 1.0))
			self.canopy_loss_weight = float(canopy_config.get("weight", 1.0))
			self.segmentation_loss_weight = float(mask_config.get("weight", 5.0))
			self.energy_loss_weight = float(energy_config.get("weight", 1.0))
			self.huber_delta = float(surface_config.get("delta", canopy_config.get("delta", energy_config.get("delta", self.huber_delta))))
			self.cawfe_mask_bce_weight = float(mask_config.get("bce_weight", 1.0))
			self.cawfe_mask_dice_weight = float(mask_config.get("dice_weight", 1.0))
			self.aux_fire_support_enabled = bool(aux_config.get("enabled", True))
			self.aux_fire_support_weight = float(aux_config.get("weight", 0.2))
		else:
			self.cawfe_mask_bce_weight = 1.0
			self.cawfe_mask_dice_weight = 1.0
			self.aux_fire_support_enabled = False
			self.aux_fire_support_weight = 0.0

	def _energy_threshold_in_target_space(self) -> float:
		"""Convert the physical active threshold into target space."""

		if str(self.energy_release_config.get("target_transform", "log1p")) == "log1p":
			return float(math.log1p(max(self.energy_active_threshold_MW, 0.0)))
		return float(self.energy_active_threshold_MW)

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

	def _energy_loss(self, pred_energy: torch.Tensor, true_energy: torch.Tensor) -> torch.Tensor:
		"""Compute the energy release regression loss."""

		if pred_energy.shape != true_energy.shape:
			raise ValueError(
				f"Energy loss expects matching shapes, got {tuple(pred_energy.shape)} and {tuple(true_energy.shape)}."
			)
		if self.energy_loss_space != "log":
			raise ValueError(
				"Unsupported multitask.energy_loss_space. "
				f"Expected 'log', got {self.energy_loss_space!r}."
			)

		threshold = self._energy_threshold_in_target_space()
		active_mask = true_energy > threshold
		weights = _build_weight_map(active_mask, self.energy_active_weight, self.energy_background_weight).to(
			device=pred_energy.device,
			dtype=pred_energy.dtype,
		)
		if self.energy_loss_name == "mse":
			loss_map = (pred_energy - true_energy) ** 2
		elif self.energy_loss_name == "huber":
			loss_map = F.smooth_l1_loss(pred_energy, true_energy, reduction="none", beta=self.huber_delta)
		else:
			raise ValueError(
				"Unsupported multitask.energy_loss. "
				f"Expected 'huber' or 'mse', got {self.energy_loss_name!r}."
			)
		return _weighted_mean(loss_map, weights)

	def _background_suppression_loss(
		self,
		pred_surface: torch.Tensor,
		pred_canopy: torch.Tensor,
		pred_mask_logits: torch.Tensor,
		pred_energy_log: torch.Tensor,
		true_surface: torch.Tensor,
		true_canopy: torch.Tensor,
		true_mask: torch.Tensor,
		true_energy_log: torch.Tensor,
	) -> torch.Tensor:
		if not self.background_suppression_enabled:
			return torch.zeros((), dtype=pred_surface.dtype, device=pred_surface.device)
		if self.background_suppression_inactive_definition == "mask_only":
			inactive = true_mask <= self.background_suppression_mask_threshold
		else:
			active = (
				(true_mask > self.background_suppression_mask_threshold)
				| (true_surface > self.background_suppression_consumed_threshold)
				| (true_canopy > self.background_suppression_consumed_threshold)
				| (true_energy_log > self.background_suppression_energy_log_threshold)
			)
			inactive = torch.logical_not(active)
		if not bool(inactive.any().item()):
			return torch.zeros((), dtype=pred_surface.dtype, device=pred_surface.device)
		terms = []
		if self.background_suppression_include_surface:
			terms.append(F.relu(pred_surface[inactive]).mean())
		if self.background_suppression_include_canopy:
			terms.append(F.relu(pred_canopy[inactive]).mean())
		if self.background_suppression_include_energy:
			terms.append(F.relu(pred_energy_log[inactive]).mean())
		if self.background_suppression_include_mask_prob:
			terms.append(torch.sigmoid(pred_mask_logits[inactive]).mean())
		if not terms:
			return torch.zeros((), dtype=pred_surface.dtype, device=pred_surface.device)
		return torch.stack(terms).mean()


	def _bce_dice_parts(self, pred_logits: torch.Tensor, true_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		if pred_logits.shape != true_mask.shape:
			raise ValueError(f"BCE+Dice expects matching shapes, got {tuple(pred_logits.shape)} and {tuple(true_mask.shape)}.")
		bce = F.binary_cross_entropy_with_logits(pred_logits, true_mask)
		dice = DiceLoss(from_logits=True, eps=self.dice_eps)(pred_logits, true_mask)
		total = self.cawfe_mask_bce_weight * bce + self.cawfe_mask_dice_weight * dice
		return bce, dice, total

	def _cawfe_huber(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
		if y_pred.shape != y_true.shape:
			raise ValueError(f"Huber expects matching shapes, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}.")
		return F.smooth_l1_loss(y_pred, y_true, beta=self.huber_delta)

	def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict[str, torch.Tensor]:
		model_output = y_pred
		aux_outputs = extract_aux_outputs(model_output)
		y_pred = extract_prediction(model_output)
		if y_pred.shape != y_true.shape:
			raise ValueError(f"MultiTaskLoss expects matching shapes, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}.")
		if y_pred.ndim != 4 or y_pred.shape[1] != self.expected_channels:
			raise ValueError(
				f"MultiTaskLoss expects tensors shaped (B, {self.expected_channels}, H, W), got {tuple(y_pred.shape)}."
			)

		pred_surface_consumed = y_pred[:, 0:1]
		true_surface_consumed = y_true[:, 0:1]
		pred_canopy_consumed = y_pred[:, 1:2]
		true_canopy_consumed = y_true[:, 1:2]
		pred_mask_logits = y_pred[:, 2:3]
		true_mask = y_true[:, 2:3]
		if not torch.isfinite(true_mask).all() or bool(((true_mask < 0) | (true_mask > 1)).any()):
			raise ValueError("Processed fire-mask targets must be finite floats in [0, 1].")
		pred_energy_log = y_pred[:, 3:4] if self.energy_output_names else torch.zeros_like(pred_surface_consumed)
		true_energy_log = y_true[:, 3:4] if self.energy_output_names else torch.zeros_like(true_surface_consumed)

		if self.cawfe_latte_loss_enabled:
			surface_loss = self._cawfe_huber(pred_surface_consumed, true_surface_consumed)
			canopy_loss = self._cawfe_huber(pred_canopy_consumed, true_canopy_consumed)
			mask_bce, mask_dice, mask_loss = self._bce_dice_parts(pred_mask_logits, true_mask)
			energy_loss = self._cawfe_huber(pred_energy_log, true_energy_log) if self.energy_output_names else torch.zeros((), dtype=y_pred.dtype, device=y_pred.device)
			aux_logits = aux_outputs.get("aux_fire_support_logits")
			aux_bce = torch.zeros((), dtype=y_pred.dtype, device=y_pred.device)
			aux_dice = torch.zeros((), dtype=y_pred.dtype, device=y_pred.device)
			aux_total = torch.zeros((), dtype=y_pred.dtype, device=y_pred.device)
			if self.aux_fire_support_enabled and torch.is_tensor(aux_logits):
				aux_bce, aux_dice, aux_total = self._bce_dice_parts(aux_logits, true_mask)
			weighted_surface = self.surface_loss_weight * surface_loss
			weighted_canopy = self.canopy_loss_weight * canopy_loss
			weighted_mask = self.segmentation_loss_weight * mask_loss
			weighted_energy = self.energy_loss_weight * energy_loss
			weighted_aux = self.aux_fire_support_weight * aux_total
			total_loss = weighted_surface + weighted_canopy + weighted_mask + weighted_energy + weighted_aux
			return {
				"total_loss": total_loss,
				"loss_total": total_loss,
				"loss_surface": surface_loss,
				"loss_canopy": canopy_loss,
				"loss_mask_bce": mask_bce,
				"loss_mask_dice": mask_dice,
				"loss_mask_total": mask_loss,
				"loss_segmentation": mask_loss,
				"loss_energy": energy_loss,
				"loss_aux_fire_support_bce": aux_bce,
				"loss_aux_fire_support_dice": aux_dice,
				"loss_aux_fire_support_total": aux_total,
				"weighted_surface": weighted_surface,
				"weighted_canopy": weighted_canopy,
				"weighted_mask": weighted_mask,
				"weighted_energy": weighted_energy,
				"weighted_aux_fire_support": weighted_aux,
			}

		surface_loss = self._regression_loss(pred_surface_consumed, true_surface_consumed, true_mask)
		canopy_loss = self._regression_loss(pred_canopy_consumed, true_canopy_consumed, true_mask)
		mask_loss = self._mask_loss(pred_mask_logits, true_mask)
		energy_loss = torch.zeros((), dtype=y_pred.dtype, device=y_pred.device)
		if self.energy_output_names:
			energy_losses = []
			for channel_offset, _ in enumerate(self.energy_output_names):
				channel_index = 3 + channel_offset
				energy_losses.append(self._energy_loss(y_pred[:, channel_index : channel_index + 1], y_true[:, channel_index : channel_index + 1]))
			energy_loss = torch.stack(energy_losses).mean()
		background_suppression_loss = self._background_suppression_loss(
			pred_surface_consumed,
			pred_canopy_consumed,
			pred_mask_logits,
			pred_energy_log,
			true_surface_consumed,
			true_canopy_consumed,
			true_mask,
			true_energy_log,
		)
		background_suppression_weighted = self.background_suppression_weight * background_suppression_loss
		total_loss = (
			self.surface_loss_weight * surface_loss
			+ self.canopy_loss_weight * canopy_loss
			+ self.segmentation_loss_weight * mask_loss
			+ self.energy_loss_weight * energy_loss
			+ background_suppression_weighted
		)
		return {
			"total_loss": total_loss,
			"loss_surface": surface_loss,
			"loss_canopy": canopy_loss,
			"loss_segmentation": mask_loss,
			"loss_energy": energy_loss,
			"background_suppression_loss": background_suppression_loss,
			"background_suppression_weighted": background_suppression_weighted,
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
