"""Canonical target-mask fire classification shared by audits and evaluation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - PyTorch is required for training only
    torch = None


FIRE_MASK_CHANNEL = 2
FIRE_MASK_THRESHOLD = 0.5
ACTIVE_FRACTION_THRESHOLD = 0.0
PREDICTED_FIRE_THRESHOLD = 0.5
ACTIVE_FRACTION_BIN_NAMES = (
    "no_fire",
    "tiny_fire",
    "small_fire",
    "medium_fire",
    "large_fire",
)
ACTIVE_FRACTION_BOUNDARIES = (0.001, 0.01, 0.05)


def classify_fire_masks(
    fire_masks: Any,
    *,
    fire_threshold: float = FIRE_MASK_THRESHOLD,
    active_fraction_threshold: float = ACTIVE_FRACTION_THRESHOLD,
) -> dict[str, Any]:
    """Classify a batch of ``(B, H, W)`` target masks canonically.

    Fire pixels are strictly greater than ``fire_threshold``. A patch is fire
    when its active fraction is greater than zero for the default threshold,
    or greater than/equal to a configured positive active-fraction threshold.
    The input may be either a NumPy array or a torch tensor; outputs use the
    same backend.
    """

    if getattr(fire_masks, "ndim", None) != 3:
        raise ValueError(f"Expected fire masks shaped (B, H, W), got {getattr(fire_masks, 'shape', None)}.")
    if int(fire_masks.shape[-2]) <= 0 or int(fire_masks.shape[-1]) <= 0:
        raise ValueError(f"Fire masks must have positive spatial dimensions, got {tuple(fire_masks.shape)}.")
    total_pixel_count = int(fire_masks.shape[-2] * fire_masks.shape[-1])
    minimum_fraction = float(active_fraction_threshold)

    if torch is not None and torch.is_tensor(fire_masks):
        if not torch.isfinite(fire_masks).all():
            raise ValueError("Target fire masks contain NaN or Inf.")
        active_pixels = (fire_masks > float(fire_threshold)).flatten(1).sum(dim=1)
        total_pixels = torch.full_like(active_pixels, total_pixel_count)
        active_fraction = active_pixels.to(dtype=torch.float64) / float(total_pixel_count)
        has_fire = active_fraction > 0.0 if minimum_fraction <= 0.0 else active_fraction >= minimum_fraction
    else:
        masks = np.asarray(fire_masks)
        if not np.isfinite(masks).all():
            raise ValueError("Target fire masks contain NaN or Inf.")
        active_pixels = (masks > float(fire_threshold)).reshape(masks.shape[0], -1).sum(axis=1, dtype=np.int64)
        total_pixels = np.full(active_pixels.shape, total_pixel_count, dtype=np.int64)
        active_fraction = active_pixels.astype(np.float64) / float(total_pixel_count)
        has_fire = active_fraction > 0.0 if minimum_fraction <= 0.0 else active_fraction >= minimum_fraction
    return {
        "active_pixels": active_pixels,
        "total_pixels": total_pixels,
        "active_fraction": active_fraction,
        "has_fire": has_fire,
        "no_fire": ~has_fire,
    }


def classify_patch_fire_state(
    fire_mask: Any,
    patch: Mapping[str, int],
    *,
    fire_threshold: float = FIRE_MASK_THRESHOLD,
    active_fraction_threshold: float = ACTIVE_FRACTION_THRESHOLD,
) -> dict[str, Any]:
    """Crop one patch from a 2-D target mask and classify it canonically."""

    mask = np.asarray(fire_mask)
    if mask.ndim != 2:
        raise ValueError(f"Fire mask must be 2-D, got shape={mask.shape}")
    y0, x0, height, width = (int(patch[key]) for key in ("y0", "x0", "height", "width"))
    if y0 < 0 or x0 < 0 or height <= 0 or width <= 0:
        raise ValueError(f"Invalid patch geometry: {dict(patch)}")
    if y0 + height > mask.shape[0] or x0 + width > mask.shape[1]:
        raise ValueError(f"Patch {dict(patch)} is outside fire mask shape={mask.shape}")
    classified = classify_fire_masks(
        mask[y0 : y0 + height, x0 : x0 + width][None, ...],
        fire_threshold=fire_threshold,
        active_fraction_threshold=active_fraction_threshold,
    )
    active_fraction = float(classified["active_fraction"][0])
    return {
        "active_pixels": int(classified["active_pixels"][0]),
        "total_pixels": int(classified["total_pixels"][0]),
        "active_fraction": active_fraction,
        "has_fire": bool(classified["has_fire"][0]),
        "no_fire": bool(classified["no_fire"][0]),
        "bin": active_fraction_bin_name(active_fraction),
    }


def active_fraction_bin_index(active_fraction: float) -> int:
    """Return the canonical class index for one patch active fraction."""
    value = float(active_fraction)
    if value == 0.0:
        return 0
    tiny_upper, small_upper, medium_upper = ACTIVE_FRACTION_BOUNDARIES
    if value < tiny_upper:
        return 1
    if value < small_upper:
        return 2
    if value < medium_upper:
        return 3
    return 4


def active_fraction_bin_name(active_fraction: float) -> str:
    """Return the canonical class name for one patch active fraction."""
    return ACTIVE_FRACTION_BIN_NAMES[active_fraction_bin_index(active_fraction)]


__all__ = [
    "ACTIVE_FRACTION_THRESHOLD",
    "ACTIVE_FRACTION_BIN_NAMES",
    "ACTIVE_FRACTION_BOUNDARIES",
    "FIRE_MASK_CHANNEL",
    "FIRE_MASK_THRESHOLD",
    "PREDICTED_FIRE_THRESHOLD",
    "active_fraction_bin_index",
    "active_fraction_bin_name",
    "classify_fire_masks",
    "classify_patch_fire_state",
]
