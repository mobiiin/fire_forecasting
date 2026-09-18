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
from src.evaluation.fire_activity import ACTIVE_FRACTION_BIN_NAMES, ACTIVE_FRACTION_BOUNDARIES



from src.training.model_outputs import extract_aux_outputs, extract_prediction, patch_fire_presence_target

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


def active_fraction_values(y_true: torch.Tensor) -> torch.Tensor:
	"""Return per-patch active fractions from canonical target mask channel 2."""
	if y_true.ndim != 4 or int(y_true.shape[1]) < 3:
		raise ValueError(f"Activity labels expect B x C x H x W targets with C >= 3, got {tuple(y_true.shape)}.")
	return (y_true[:, 2] > 0.5).to(dtype=torch.float32).flatten(1).mean(dim=1)


def active_fraction_class_labels(y_true: torch.Tensor) -> torch.Tensor:
	"""Canonical no/tiny/small/medium/large labels from target-mask activity."""
	fraction = active_fraction_values(y_true)
	tiny_upper, small_upper, medium_upper = ACTIVE_FRACTION_BOUNDARIES
	labels = torch.zeros_like(fraction, dtype=torch.long)
	labels[(fraction > 0.0) & (fraction < tiny_upper)] = 1
	labels[(fraction >= tiny_upper) & (fraction < small_upper)] = 2
	labels[(fraction >= small_upper) & (fraction < medium_upper)] = 3
	labels[fraction >= medium_upper] = 4
	return labels


def supervised_contrastive_positive_mask(labels: torch.Tensor) -> torch.Tensor:
	"""Same-class positive pairs with every anchor's self-pair excluded."""
	if labels.ndim != 1:
		raise ValueError(f"Contrastive labels must be one-dimensional, got {tuple(labels.shape)}.")
	self_mask = torch.eye(int(labels.shape[0]), dtype=torch.bool, device=labels.device)
	return labels[:, None].eq(labels[None, :]) & ~self_mask


def rbf_mmd(x: torch.Tensor, y: torch.Tensor, bandwidths: list[float] | tuple[float, ...]) -> torch.Tensor:
	"""Biased multi-bandwidth RBF MMD; identical samples evaluate to zero."""
	if x.ndim != 2 or y.ndim != 2 or int(x.shape[1]) != int(y.shape[1]):
		raise ValueError(f"MMD expects N x D and M x D features, got {tuple(x.shape)} and {tuple(y.shape)}.")
	if not bandwidths or any(float(value) <= 0.0 for value in bandwidths):
		raise ValueError("fire_mmd.bandwidths must contain positive values.")
	def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
		distance = torch.cdist(left.float(), right.float(), p=2).square()
		terms = [torch.exp(-distance / (2.0 * float(bandwidth) ** 2)) for bandwidth in bandwidths]
		return torch.stack(terms, dim=0).mean(dim=0).to(dtype=left.dtype)
	value = kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean()
	return torch.clamp(value, min=0.0)


def cross_fire_mmd(
	features: torch.Tensor,
	domain_labels: torch.Tensor,
	bandwidths: list[float] | tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Average MMD across all domain pairs represented in a batch."""
	if features.ndim != 2 or domain_labels.ndim != 1 or int(features.shape[0]) != int(domain_labels.shape[0]):
		raise ValueError("Cross-fire MMD expects N x D features and N labels.")
	unique = torch.unique(domain_labels)
	pair_losses = []
	for left_index in range(int(unique.numel())):
		for right_index in range(left_index + 1, int(unique.numel())):
			pair_losses.append(
				rbf_mmd(features[domain_labels == unique[left_index]], features[domain_labels == unique[right_index]], bandwidths)
			)
	if not pair_losses:
		zero = features.sum() * 0.0
		return zero, zero.detach()
	return torch.stack(pair_losses).mean(), features.new_tensor(1.0)


def supervised_contrastive_loss(
	embeddings: torch.Tensor,
	labels: torch.Tensor,
	temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Supervised contrastive loss over anchors that have an in-batch positive."""
	if embeddings.ndim != 2 or labels.ndim != 1 or int(embeddings.shape[0]) != int(labels.shape[0]):
		raise ValueError("Supervised contrastive loss expects N x D embeddings and N labels.")
	if float(temperature) <= 0.0:
		raise ValueError("supervised_contrastive.temperature must be positive.")
	count = int(embeddings.shape[0])
	if count < 2:
		zero = embeddings.sum() * 0.0
		return zero, zero.detach()
	normalized = F.normalize(embeddings, p=2, dim=1)
	logits = normalized @ normalized.transpose(0, 1) / float(temperature)
	self_mask = torch.eye(count, dtype=torch.bool, device=embeddings.device)
	logits = logits.masked_fill(self_mask, float("-inf"))
	positive_mask = supervised_contrastive_positive_mask(labels)
	valid = positive_mask.any(dim=1)
	valid_fraction = valid.to(dtype=embeddings.dtype).mean()
	if not bool(valid.any().item()):
		return embeddings.sum() * 0.0, valid_fraction
	log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
	positive_count = positive_mask.sum(dim=1).clamp(min=1)
	per_anchor = -(log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1) / positive_count)
	return per_anchor[valid].mean(), valid_fraction


def physical_state_targets(y_true: torch.Tensor) -> torch.Tensor:
	"""Derive active fraction and active-only canopy/energy means from targets."""
	if y_true.ndim != 4 or int(y_true.shape[1]) < 4:
		raise ValueError(f"Physical-state targets expect B x C x H x W with C >= 4, got {tuple(y_true.shape)}.")
	active = y_true[:, 2] > 0.5
	active_float = active.to(dtype=y_true.dtype)
	count = active_float.flatten(1).sum(dim=1)
	denominator = count.clamp(min=1.0)
	fraction = active_float.flatten(1).mean(dim=1)
	active_canopy_mean = (y_true[:, 1] * active_float).flatten(1).sum(dim=1) / denominator
	canopy_state = torch.log1p(torch.clamp(active_canopy_mean, min=0.0))
	energy_state = (y_true[:, 3] * active_float).flatten(1).sum(dim=1) / denominator
	return torch.stack([fraction, canopy_state, energy_state], dim=1)


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
			patch_fire_config = _get_section(self.training_loss_config, "patch_fire")
			self.patch_fire_enabled = bool(patch_fire_config.get("enabled", False))
			self.patch_fire_weight = float(patch_fire_config.get("weight", 0.2))
			self.surface_loss_weight = float(surface_config.get("weight", 1.0))
			self.canopy_loss_weight = float(canopy_config.get("weight", 1.0))
			self.segmentation_loss_weight = float(mask_config.get("weight", 5.0))
			self.energy_loss_weight = float(energy_config.get("weight", 1.0))
			self.huber_delta = float(surface_config.get("delta", canopy_config.get("delta", energy_config.get("delta", self.huber_delta))))
			self.cawfe_mask_bce_weight = float(mask_config.get("bce_weight", 1.0))
			self.cawfe_mask_dice_weight = float(mask_config.get("dice_weight", 1.0))
			self.aux_fire_support_enabled = bool(aux_config.get("enabled", True))
			self.aux_fire_support_weight = float(aux_config.get("weight", 0.2))
			self.aux_fire_support_target = str(aux_config.get("target", "original_mask")).lower()
			self.aux_fire_support_dilation_radius = int(aux_config.get("dilation_radius", 0))
			if self.aux_fire_support_target not in {"original_mask", "dilated_mask"}: raise ValueError("auxiliary_fire_support.target must be original_mask or dilated_mask.")
		else:
			self.cawfe_mask_bce_weight = 1.0
			self.cawfe_mask_dice_weight = 1.0
			self.aux_fire_support_enabled = False
			self.aux_fire_support_weight = 0.0
			self.aux_fire_support_target = "original_mask"
			self.aux_fire_support_dilation_radius = 0
			self.patch_fire_enabled = False
			self.patch_fire_weight = 0.0
		self.cawfe_config = _get_section(config, "cawfe_latte")
		def optional_cawfe_section(name: str) -> dict:
			value = self.cawfe_config.get(name)
			return dict(value) if isinstance(value, dict) else {}
		self.domain_adversarial_config = optional_cawfe_section("domain_adversarial")
		self.fire_mmd_config = optional_cawfe_section("fire_mmd")
		self.mask_guidance_config = optional_cawfe_section("mask_guided_regression")
		self.regression_moe_config = optional_cawfe_section("regression_moe")
		self.supervised_contrastive_config = optional_cawfe_section("supervised_contrastive")
		self.physical_state_aux_config = optional_cawfe_section("physical_state_aux")

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



	def _support_target(self, true_mask: torch.Tensor) -> torch.Tensor:
		"""Use saved masks directly or derive a dilation only for the auxiliary loss."""
		if self.aux_fire_support_target == "original_mask" or self.aux_fire_support_dilation_radius <= 0:
			return true_mask
		radius = self.aux_fire_support_dilation_radius
		return F.max_pool2d(true_mask, kernel_size=2 * radius + 1, stride=1, padding=radius)

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
			aux_logits = aux_outputs.get("support_logits", aux_outputs.get("aux_fire_support_logits"))
			aux_bce = torch.zeros((), dtype=y_pred.dtype, device=y_pred.device)
			aux_dice = torch.zeros((), dtype=y_pred.dtype, device=y_pred.device)
			aux_total = torch.zeros((), dtype=y_pred.dtype, device=y_pred.device)
			if self.aux_fire_support_enabled and torch.is_tensor(aux_logits):
				aux_bce, aux_dice, aux_total = self._bce_dice_parts(aux_logits, self._support_target(true_mask))
			weighted_surface = self.surface_loss_weight * surface_loss
			weighted_canopy = self.canopy_loss_weight * canopy_loss
			weighted_mask = self.segmentation_loss_weight * mask_loss
			weighted_energy = self.energy_loss_weight * energy_loss
			weighted_aux = self.aux_fire_support_weight * aux_total
			patch_fire_loss = torch.zeros((), dtype=y_pred.dtype, device=y_pred.device)
			weighted_patch_fire = torch.zeros((), dtype=y_pred.dtype, device=y_pred.device)
			if self.patch_fire_enabled:
				patch_fire_logit = aux_outputs.get("patch_fire_logit")
				if not torch.is_tensor(patch_fire_logit):
					raise ValueError("training.loss.patch_fire.enabled=true requires model output 'patch_fire_logit'.")
				patch_target = patch_fire_presence_target(y_true).to(device=patch_fire_logit.device, dtype=patch_fire_logit.dtype)
				if patch_fire_logit.shape != patch_target.shape:
					raise ValueError(f"Patch-fire logits and targets must match, got {tuple(patch_fire_logit.shape)} and {tuple(patch_target.shape)}.")
				patch_fire_loss = F.binary_cross_entropy_with_logits(patch_fire_logit, patch_target)
				weighted_patch_fire = self.patch_fire_weight * patch_fire_loss

			zero = y_pred.sum() * 0.0
			auxiliary_total = zero
			diagnostics: dict[str, torch.Tensor] = {}
			alpha = aux_outputs.get("mask_guidance_alpha")
			if torch.is_tensor(alpha):
				diagnostics["mask_guidance_alpha"] = alpha
			guidance = aux_outputs.get("mask_guidance_attention")
			if torch.is_tensor(guidance):
				diagnostics["guidance_mean"] = guidance.mean()
				diagnostics["guidance_std"] = guidance.std(unbiased=False)
				active_pixels = true_mask > 0.5
				if bool(active_pixels.any().item()):
					diagnostics["guidance_mean_active"] = guidance[active_pixels].mean()
				if bool((~active_pixels).any().item()):
					diagnostics["guidance_mean_inactive"] = guidance[~active_pixels].mean()

			predicted_state = aux_outputs.get("physical_state_prediction")
			state_loss = None
			if torch.is_tensor(predicted_state):
				state_target = physical_state_targets(y_true).to(dtype=predicted_state.dtype)
				state_losses = F.smooth_l1_loss(predicted_state, state_target, reduction="none").mean(dim=0)
				state_maes = torch.abs(predicted_state - state_target).mean(dim=0)
				state_loss = state_losses.mean()
				diagnostics.update(
					physical_state_aux_loss=state_loss,
					physical_active_fraction_loss=state_losses[0],
					physical_canopy_state_loss=state_losses[1],
					physical_energy_state_loss=state_losses[2],
					physical_active_fraction_mae=state_maes[0],
					physical_canopy_state_mae=state_maes[1],
					physical_energy_state_mae=state_maes[2],
				)

			auxiliary_training = bool(aux_outputs.get("auxiliary_training", False))
			if auxiliary_training:
				domain_labels = aux_outputs.get("fire_domain_labels")
				if bool(self.domain_adversarial_config.get("enabled", False)):
					domain_logits = aux_outputs.get("domain_logits")
					if not torch.is_tensor(domain_logits) or not torch.is_tensor(domain_labels):
						raise ValueError("Domain adversarial training requires domain_logits and training fire_domain_labels.")
					domain_loss = F.cross_entropy(domain_logits, domain_labels.long())
					domain_accuracy = (domain_logits.argmax(dim=1) == domain_labels.long()).to(dtype=y_pred.dtype).mean()
					domain_random_chance = y_pred.new_tensor(1.0 / float(domain_logits.shape[1]))
					auxiliary_total = auxiliary_total + float(self.domain_adversarial_config.get("loss_weight", 0.05)) * domain_loss
					diagnostics.update(domain_loss=domain_loss, domain_accuracy=domain_accuracy, domain_random_chance=domain_random_chance)
				if bool(self.fire_mmd_config.get("enabled", False)):
					mmd_features = aux_outputs.get("fire_mmd_features")
					if not torch.is_tensor(mmd_features) or not torch.is_tensor(domain_labels):
						raise ValueError("Fire MMD training requires fire_mmd_features and training fire_domain_labels.")
					mmd_loss, valid_fraction = cross_fire_mmd(mmd_features, domain_labels.long(), list(self.fire_mmd_config.get("bandwidths", [0.5, 1.0, 2.0, 4.0])))
					auxiliary_total = auxiliary_total + float(self.fire_mmd_config.get("loss_weight", 0.05)) * mmd_loss
					diagnostics.update(mmd_loss=mmd_loss, mmd_valid_batch_fraction=valid_fraction)
				if bool(self.regression_moe_config.get("enabled", False)):
					router_weights = aux_outputs.get("router_weights")
					if not torch.is_tensor(router_weights):
						raise ValueError("Regression MoE training requires router_weights.")
					mean_weights = router_weights.mean(dim=0)
					uniform = torch.full_like(mean_weights, 1.0 / float(mean_weights.numel()))
					load_balance_loss = torch.mean((mean_weights - uniform).square())
					router_entropy = -(router_weights * torch.log(router_weights.clamp(min=1.0e-8))).sum(dim=1).mean()
					auxiliary_total = auxiliary_total + float(self.regression_moe_config.get("load_balance_weight", 0.01)) * load_balance_loss
					diagnostics["load_balance_loss"] = load_balance_loss
					diagnostics["router_entropy"] = router_entropy
					diagnostics["router_max_probability_mean"] = router_weights.max(dim=1).values.mean()
					for expert_index, mean_weight in enumerate(mean_weights):
						diagnostics[f"expert_{expert_index + 1}_mean_weight"] = mean_weight
					activity_labels = active_fraction_class_labels(y_true)
					for class_index, class_name in enumerate(ACTIVE_FRACTION_BIN_NAMES):
						class_samples = activity_labels == class_index
						if bool(class_samples.any().item()):
							class_weights = router_weights[class_samples].mean(dim=0)
							for expert_index, mean_weight in enumerate(class_weights):
								diagnostics[f"router_{class_name}_expert_{expert_index + 1}_mean_weight"] = mean_weight
				if bool(self.supervised_contrastive_config.get("enabled", False)):
					embedding = aux_outputs.get("contrastive_embedding")
					if not torch.is_tensor(embedding):
						raise ValueError("Supervised contrastive training requires contrastive_embedding.")
					activity_labels = active_fraction_class_labels(y_true)
					contrastive_loss, valid_fraction = supervised_contrastive_loss(embedding, activity_labels, float(self.supervised_contrastive_config.get("temperature", 0.1)))
					auxiliary_total = auxiliary_total + float(self.supervised_contrastive_config.get("loss_weight", 0.05)) * contrastive_loss
					diagnostics.update(contrastive_loss=contrastive_loss, valid_contrastive_anchor_fraction=valid_fraction)
					diagnostic_class_names = ("no_fire", "tiny", "small", "medium", "large")
					for class_index, class_name in enumerate(diagnostic_class_names):
						diagnostics[f"batch_fraction_{class_name}"] = (activity_labels == class_index).to(dtype=y_pred.dtype).mean()
					cosine = embedding @ embedding.transpose(0, 1)
					positive_mask = supervised_contrastive_positive_mask(activity_labels)
					different_mask = activity_labels[:, None].ne(activity_labels[None, :])
					diagnostics["same_class_cosine_similarity"] = cosine[positive_mask].mean() if bool(positive_mask.any().item()) else zero
					diagnostics["different_class_cosine_similarity"] = cosine[different_mask].mean() if bool(different_mask.any().item()) else zero
				if bool(self.physical_state_aux_config.get("enabled", False)):
					if state_loss is None:
						raise ValueError("Physical-state auxiliary training requires physical_state_prediction.")
					auxiliary_total = auxiliary_total + float(self.physical_state_aux_config.get("loss_weight", 0.05)) * state_loss
			total_loss = weighted_surface + weighted_canopy + weighted_mask + weighted_energy + weighted_aux + weighted_patch_fire + auxiliary_total
			result = {
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
			if self.patch_fire_enabled:
				result["loss_patch_fire_bce"] = patch_fire_loss
				result["loss_patch_fire_total"] = patch_fire_loss
				result["weighted_patch_fire"] = weighted_patch_fire
			result.update(diagnostics)
			return result

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
