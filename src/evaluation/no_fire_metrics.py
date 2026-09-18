"""Canonical target-defined no-fire classification and post-hoc metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.evaluation.fire_activity import (
    ACTIVE_FRACTION_THRESHOLD,
    FIRE_MASK_CHANNEL,
    FIRE_MASK_THRESHOLD,
    PREDICTED_FIRE_THRESHOLD,
    classify_fire_masks,
)


def classify_target_masks(
    targets: torch.Tensor,
    *,
    mask_channel: int = FIRE_MASK_CHANNEL,
    fire_threshold: float = FIRE_MASK_THRESHOLD,
    active_fraction_threshold: float = ACTIVE_FRACTION_THRESHOLD,
) -> dict[str, torch.Tensor]:
    """Classify patches exactly like ``check_patch_fire_balance.classify_patch``.

    A patch has fire iff at least one target-mask pixel is strictly greater than
    ``fire_threshold``. Continuous targets and model predictions never affect
    this classification.
    """

    if targets.ndim != 4:
        raise ValueError(f"Expected targets shaped (B, C, H, W), got {tuple(targets.shape)}.")
    if int(mask_channel) < 0 or int(mask_channel) >= int(targets.shape[1]):
        raise ValueError(f"mask_channel={mask_channel} is outside C={targets.shape[1]}.")
    return classify_fire_masks(
        targets[:, int(mask_channel)],
        fire_threshold=fire_threshold,
        active_fraction_threshold=active_fraction_threshold,
    )


@dataclass
class FullValidationNoFireAccumulator:
    """Stream full-validation no-fire metrics without retaining predictions."""

    fire_threshold: float = FIRE_MASK_THRESHOLD
    prediction_threshold: float = PREDICTED_FIRE_THRESHOLD
    active_fraction_threshold: float = ACTIVE_FRACTION_THRESHOLD
    total_patches: int = 0
    fire_patches: int = 0
    no_fire_patches: int = 0
    no_fire_pixels: int = 0
    mask_probability_sum: float = 0.0
    false_positive_pixels: int = 0
    false_positive_patches: int = 0
    surface_sum: float = 0.0
    canopy_sum: float = 0.0
    energy_log_sum: float = 0.0
    energy_mw_sum: float = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.ndim != 4 or target.ndim != 4:
            raise ValueError("Prediction and target must both have shape (B, C, H, W).")
        if prediction.shape != target.shape:
            raise ValueError(f"Prediction/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}.")
        if prediction.shape[1] < 4:
            raise ValueError(f"Expected at least four multitask channels, got C={prediction.shape[1]}.")
        classification = classify_target_masks(
            target,
            fire_threshold=self.fire_threshold,
            active_fraction_threshold=self.active_fraction_threshold,
        )
        no_fire = classification["no_fire"]
        batch_size = int(target.shape[0])
        batch_no_fire = int(no_fire.sum().item())
        self.total_patches += batch_size
        self.no_fire_patches += batch_no_fire
        self.fire_patches += batch_size - batch_no_fire
        if batch_no_fire == 0:
            return

        selected = prediction[no_fire].detach().to(dtype=torch.float32)
        if not torch.isfinite(selected).all():
            raise ValueError("Model predictions contain NaN or Inf on target no-fire patches.")
        mask_probability = torch.sigmoid(selected[:, 2])
        predicted_fire = mask_probability > float(self.prediction_threshold)
        pixel_count = int(mask_probability.numel())
        self.no_fire_pixels += pixel_count
        self.mask_probability_sum += float(mask_probability.sum(dtype=torch.float64).item())
        self.false_positive_pixels += int(predicted_fire.sum().item())
        self.false_positive_patches += int(predicted_fire.flatten(1).any(dim=1).sum().item())
        self.surface_sum += float(selected[:, 0].sum(dtype=torch.float64).item())
        self.canopy_sum += float(selected[:, 1].sum(dtype=torch.float64).item())
        self.energy_log_sum += float(selected[:, 3].sum(dtype=torch.float64).item())
        energy_mw = torch.clamp(torch.expm1(selected[:, 3].to(dtype=torch.float64)), min=0.0)
        self.energy_mw_sum += float(energy_mw.sum(dtype=torch.float64).item())

    def finalize(self) -> dict[str, Any]:
        if self.total_patches != self.fire_patches + self.no_fire_patches:
            raise RuntimeError("Internal patch counts do not sum to the evaluated total.")
        if self.no_fire_patches == 0:
            means = {
                "full_val_no_fire_mask_prob_mean": None,
                "full_val_no_fire_mask_false_positive_rate": None,
                "full_val_no_fire_patch_false_positive_rate": None,
                "full_val_no_fire_surface_pred_mean": None,
                "full_val_no_fire_canopy_pred_mean": None,
                "full_val_no_fire_energy_log_pred_mean": None,
                "full_val_no_fire_energy_mw_pred_mean": None,
                "full_val_no_fire_energy_MW_pred_mean": None,
            }
        else:
            if self.no_fire_pixels <= 0:
                raise RuntimeError("No-fire patches were counted but no no-fire pixels were accumulated.")
            pixel_denominator = float(self.no_fire_pixels)
            means = {
                "full_val_no_fire_mask_prob_mean": self.mask_probability_sum / pixel_denominator,
                "full_val_no_fire_mask_false_positive_rate": self.false_positive_pixels / pixel_denominator,
                "full_val_no_fire_patch_false_positive_rate": self.false_positive_patches / float(self.no_fire_patches),
                "full_val_no_fire_surface_pred_mean": self.surface_sum / pixel_denominator,
                "full_val_no_fire_canopy_pred_mean": self.canopy_sum / pixel_denominator,
                "full_val_no_fire_energy_log_pred_mean": self.energy_log_sum / pixel_denominator,
                "full_val_no_fire_energy_mw_pred_mean": self.energy_mw_sum / pixel_denominator,
                "full_val_no_fire_energy_MW_pred_mean": self.energy_mw_sum / pixel_denominator,
            }
        return {
            "full_val_total_patch_count": self.total_patches,
            "full_val_fire_patch_count": self.fire_patches,
            "full_val_no_fire_patch_count": self.no_fire_patches,
            "full_val_no_fire_pixel_count": self.no_fire_pixels,
            "full_val_no_fire_false_positive_pixel_count": self.false_positive_pixels,
            "full_val_no_fire_false_positive_patch_count": self.false_positive_patches,
            "full_val_no_fire_percent": (
                100.0 * self.no_fire_patches / self.total_patches if self.total_patches else 0.0
            ),
            **means,
        }
