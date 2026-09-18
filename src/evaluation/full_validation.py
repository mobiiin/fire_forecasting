"""Exact streaming full-validation evaluation for trained forecasting models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import SequentialSampler

from src.data.dataset import metadata_batch_to_list
from src.evaluation.fire_activity import (
    ACTIVE_FRACTION_THRESHOLD,
    FIRE_MASK_CHANNEL,
    FIRE_MASK_THRESHOLD,
    PREDICTED_FIRE_THRESHOLD,
)
from src.evaluation.no_fire_metrics import FullValidationNoFireAccumulator, classify_target_masks
from src.training.batch_utils import unpack_batch
from src.training.hardware import autocast_context
from src.training.input_normalization import apply_input_normalization, build_input_normalizer_for_loader
from src.training.model_outputs import extract_prediction


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _sqrt_mean(sum_squared: float, count: int) -> float | None:
    return math.sqrt(max(0.0, float(sum_squared) / float(count))) if count > 0 else None


@dataclass
class FullValidationAccumulator:
    """Accumulate exact global segmentation/regression metrics in float64."""

    active_fraction_threshold: float = ACTIVE_FRACTION_THRESHOLD
    total_patches: int = 0
    fire_patches: int = 0
    no_fire_patches: int = 0
    total_pixels: int = 0
    active_pixels: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    surface_absolute_error: float = 0.0
    surface_squared_error: float = 0.0
    canopy_absolute_error: float = 0.0
    canopy_squared_error: float = 0.0
    energy_absolute_error: float = 0.0
    energy_squared_error: float = 0.0
    active_surface_absolute_error: float = 0.0
    active_surface_squared_error: float = 0.0
    active_canopy_absolute_error: float = 0.0
    active_canopy_squared_error: float = 0.0
    active_energy_absolute_error: float = 0.0
    active_energy_squared_error: float = 0.0
    no_fire: FullValidationNoFireAccumulator = field(default_factory=FullValidationNoFireAccumulator)

    def __post_init__(self) -> None:
        self.no_fire.active_fraction_threshold = float(self.active_fraction_threshold)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] < 4:
            raise ValueError(
                f"Full validation expects matching (B, >=4, H, W) tensors, got {tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        prediction = prediction.detach().to(dtype=torch.float64)
        target = target.detach().to(dtype=torch.float64)
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("Full-validation prediction/target contains NaN or Inf.")

        classification = classify_target_masks(
            target,
            fire_threshold=FIRE_MASK_THRESHOLD,
            active_fraction_threshold=self.active_fraction_threshold,
        )
        fire = classification["has_fire"]
        batch_size = int(target.shape[0])
        batch_fire = int(fire.sum().item())
        self.total_patches += batch_size
        self.fire_patches += batch_fire
        self.no_fire_patches += batch_size - batch_fire

        true_mask = target[:, FIRE_MASK_CHANNEL] > FIRE_MASK_THRESHOLD
        probability = torch.sigmoid(prediction[:, FIRE_MASK_CHANNEL])
        predicted_mask = probability > PREDICTED_FIRE_THRESHOLD
        self.total_pixels += int(true_mask.numel())
        self.active_pixels += int(true_mask.sum().item())
        self.true_positive += int((predicted_mask & true_mask).sum().item())
        self.false_positive += int((predicted_mask & ~true_mask).sum().item())
        self.false_negative += int((~predicted_mask & true_mask).sum().item())

        for channel, prefix in ((0, "surface"), (1, "canopy"), (3, "energy")):
            error = prediction[:, channel] - target[:, channel]
            absolute = error.abs()
            setattr(self, f"{prefix}_absolute_error", getattr(self, f"{prefix}_absolute_error") + float(absolute.sum().item()))
            setattr(self, f"{prefix}_squared_error", getattr(self, f"{prefix}_squared_error") + float(error.square().sum().item()))
            if true_mask.any():
                setattr(
                    self,
                    f"active_{prefix}_absolute_error",
                    getattr(self, f"active_{prefix}_absolute_error") + float(absolute[true_mask].sum().item()),
                )
                setattr(
                    self,
                    f"active_{prefix}_squared_error",
                    getattr(self, f"active_{prefix}_squared_error") + float(error.square()[true_mask].sum().item()),
                )
        self.no_fire.update(prediction.to(dtype=torch.float32), target.to(dtype=torch.float32))

    def finalize(self) -> dict[str, Any]:
        if self.total_patches != self.fire_patches + self.no_fire_patches:
            raise RuntimeError("Full-validation patch counts do not sum to the evaluated total.")
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        union = self.true_positive + self.false_positive + self.false_negative
        metrics: dict[str, Any] = {
            "full_val_total_patch_count": self.total_patches,
            "full_val_fire_patch_count": self.fire_patches,
            "full_val_no_fire_patch_count": self.no_fire_patches,
            "full_val_fire_patch_percent": 100.0 * self.fire_patches / self.total_patches if self.total_patches else 0.0,
            "full_val_no_fire_patch_percent": 100.0 * self.no_fire_patches / self.total_patches if self.total_patches else 0.0,
            "full_val_no_fire_percent": 100.0 * self.no_fire_patches / self.total_patches if self.total_patches else 0.0,
            "full_val_dice": _safe_divide(2 * self.true_positive, denominator),
            "full_val_iou": _safe_divide(self.true_positive, union),
            "full_val_precision": _safe_divide(self.true_positive, self.true_positive + self.false_positive),
            "full_val_recall": _safe_divide(self.true_positive, self.true_positive + self.false_negative),
            "full_val_surface_mae": _safe_divide(self.surface_absolute_error, self.total_pixels),
            "full_val_surface_rmse": _sqrt_mean(self.surface_squared_error, self.total_pixels),
            "full_val_canopy_mae": _safe_divide(self.canopy_absolute_error, self.total_pixels),
            "full_val_canopy_rmse": _sqrt_mean(self.canopy_squared_error, self.total_pixels),
            "full_val_energy_log_mae": _safe_divide(self.energy_absolute_error, self.total_pixels),
            "full_val_energy_log_rmse": _sqrt_mean(self.energy_squared_error, self.total_pixels),
            "full_val_active_surface_mae": _safe_divide(self.active_surface_absolute_error, self.active_pixels),
            "full_val_active_surface_rmse": _sqrt_mean(self.active_surface_squared_error, self.active_pixels),
            "full_val_active_canopy_mae": _safe_divide(self.active_canopy_absolute_error, self.active_pixels),
            "full_val_active_canopy_rmse": _sqrt_mean(self.active_canopy_squared_error, self.active_pixels),
            "full_val_active_energy_log_mae": _safe_divide(self.active_energy_absolute_error, self.active_pixels),
            "full_val_active_energy_log_rmse": _sqrt_mean(self.active_energy_squared_error, self.active_pixels),
            "full_val_mask_true_positive_pixel_count": self.true_positive,
            "full_val_mask_false_positive_pixel_count": self.false_positive,
            "full_val_mask_false_negative_pixel_count": self.false_negative,
            "full_val_active_pixel_count": self.active_pixels,
            "full_val_total_pixel_count": self.total_pixels,
        }
        metrics.update(self.no_fire.finalize())
        # Compatibility aliases match training metric names while keeping the
        # explicit full_val_ scope required by ablation reporting.
        metrics.update(
            {
                "full_val_mask_dice": metrics["full_val_dice"],
                "full_val_mask_iou": metrics["full_val_iou"],
                "full_val_mask_precision": metrics["full_val_precision"],
                "full_val_mask_recall": metrics["full_val_recall"],
                "full_val_surface_consumed_mae": metrics["full_val_surface_mae"],
                "full_val_surface_consumed_rmse": metrics["full_val_surface_rmse"],
                "full_val_canopy_consumed_mae": metrics["full_val_canopy_mae"],
                "full_val_canopy_consumed_rmse": metrics["full_val_canopy_rmse"],
                "full_val_active_surface_consumed_mae": metrics["full_val_active_surface_mae"],
                "full_val_active_surface_consumed_rmse": metrics["full_val_active_surface_rmse"],
                "full_val_active_canopy_consumed_mae": metrics["full_val_active_canopy_mae"],
                "full_val_active_canopy_consumed_rmse": metrics["full_val_active_canopy_rmse"],
            }
        )
        return metrics


def _per_fire_metrics(accumulator: FullValidationAccumulator) -> dict[str, Any]:
    metrics = accumulator.finalize()
    return {
        "sample_count": metrics["full_val_total_patch_count"],
        "fire_patch_count": metrics["full_val_fire_patch_count"],
        "no_fire_patch_count": metrics["full_val_no_fire_patch_count"],
        "dice": metrics["full_val_dice"],
        "iou": metrics["full_val_iou"],
        "surface_mae": metrics["full_val_surface_mae"],
        "surface_rmse": metrics["full_val_surface_rmse"],
        "canopy_mae": metrics["full_val_canopy_mae"],
        "canopy_rmse": metrics["full_val_canopy_rmse"],
        "energy_log_mae": metrics["full_val_energy_log_mae"],
        "energy_log_rmse": metrics["full_val_energy_log_rmse"],
        "active_canopy_mae": metrics["full_val_active_canopy_mae"],
        "active_energy_log_mae": metrics["full_val_active_energy_log_mae"],
        "no_fire_mask_false_positive_rate": metrics["full_val_no_fire_mask_false_positive_rate"],
        "no_fire_energy_log_pred_mean": metrics["full_val_no_fire_energy_log_pred_mean"],
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _model_call(model: torch.nn.Module, inputs: torch.Tensor, terrain: torch.Tensor | None):
    return model(inputs) if terrain is None else model(inputs, terrain=terrain)


def evaluate_full_validation(
    *,
    model: torch.nn.Module,
    val_loader: Any,
    config: Mapping[str, Any],
    device: torch.device,
    amp_dtype: Any,
    run_dir: str | Path,
    checkpoint_path: str | Path,
    checkpoint_epoch: int | None,
    expected_counts: Mapping[str, Any] | None = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Evaluate every validation sample once and persist exact run artifacts."""

    if bool(getattr(val_loader, "drop_last", False)):
        raise RuntimeError("Full validation requires drop_last=false.")
    if not isinstance(getattr(val_loader, "sampler", None), SequentialSampler):
        raise RuntimeError(f"Full validation requires deterministic no-shuffle sampling, got {type(getattr(val_loader, 'sampler', None)).__name__}.")
    expected_samples = len(val_loader.dataset)
    if expected_samples <= 0:
        raise RuntimeError("Full validation dataset is empty.")

    model.eval()
    normalizer = build_input_normalizer_for_loader(val_loader, device, int(config.get("model", {}).get("input_channels", 0)))
    overall = FullValidationAccumulator()
    per_fire: dict[str, FullValidationAccumulator] = {}
    sample_ids: set[str] = set()
    sample_id_count = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(val_loader, start=1):
            x_raw, y_raw, extra = unpack_batch(batch)
            terrain_raw = extra.get("terrain")
            x = apply_input_normalization(x_raw.to(device, non_blocking=True), normalizer)
            y = y_raw.to(device, non_blocking=True).float()
            terrain = terrain_raw.to(device, non_blocking=True) if terrain_raw is not None else None
            with autocast_context(device, amp_dtype):
                output = _model_call(model, x, terrain)
            prediction = extract_prediction(output).float()
            overall.update(prediction, y)

            metadata_batch = extra.get("metadata")
            metadata_items = metadata_batch_to_list(metadata_batch, batch_size=int(y.shape[0])) if isinstance(metadata_batch, Mapping) else []
            if metadata_items and len(metadata_items) != int(y.shape[0]):
                raise RuntimeError("Validation metadata batch length does not match prediction batch length.")
            grouped: dict[str, list[int]] = {}
            for sample_index, metadata in enumerate(metadata_items):
                fire_name = str(metadata.get("fire_name", metadata.get("fire", "unknown")))
                grouped.setdefault(fire_name, []).append(sample_index)
                sample_id = metadata.get("sample_id")
                if sample_id is not None:
                    sample_id_count += 1
                    text_id = str(sample_id)
                    if text_id in sample_ids:
                        raise RuntimeError(f"Full validation visited sample_id more than once: {text_id}")
                    sample_ids.add(text_id)
            for fire_name, indices in grouped.items():
                if fire_name not in per_fire:
                    per_fire[fire_name] = FullValidationAccumulator()
                per_fire[fire_name].update(prediction[indices], y[indices])
            if logger is not None and (batch_index == 1 or batch_index % 100 == 0 or batch_index == len(val_loader)):
                logger.info("Full validation progress: %s/%s batches", batch_index, len(val_loader))

    metrics = overall.finalize()
    evaluated = int(metrics["full_val_total_patch_count"])
    if evaluated != expected_samples:
        raise RuntimeError(f"Full validation evaluated {evaluated} samples, expected exactly {expected_samples}.")
    if sample_id_count and (sample_id_count != evaluated or len(sample_ids) != evaluated):
        raise RuntimeError(
            f"Full-validation sample-ID coverage mismatch: ids={len(sample_ids)} metadata={sample_id_count} evaluated={evaluated}."
        )
    if int(metrics["full_val_fire_patch_count"]) + int(metrics["full_val_no_fire_patch_count"]) != evaluated:
        raise RuntimeError("Full validation fire/no-fire counts do not equal the total.")
    if expected_counts is not None:
        expected_triplet = (
            int(expected_counts["total"]),
            int(expected_counts["fire"]),
            int(expected_counts["no_fire"]),
        )
        actual_triplet = (
            evaluated,
            int(metrics["full_val_fire_patch_count"]),
            int(metrics["full_val_no_fire_patch_count"]),
        )
        if expected_triplet[2] > 0 and actual_triplet[2] == 0:
            raise RuntimeError("Canonical validation scan found no-fire samples, but full evaluation counted zero.")
        if actual_triplet != expected_triplet:
            raise RuntimeError(f"Full-validation counts {actual_triplet} disagree with canonical counts {expected_triplet}.")

    per_fire_payload = {name: _per_fire_metrics(accumulator) for name, accumulator in sorted(per_fire.items())}
    if per_fire_payload and sum(int(item["sample_count"]) for item in per_fire_payload.values()) != evaluated:
        raise RuntimeError("Per-fire sample counts do not sum to the full-validation total.")
    run_path = Path(run_dir).expanduser().resolve()
    created_at = datetime.now(timezone.utc).isoformat()
    full_payload = {
        "schema_version": 1,
        "created_at": created_at,
        "metric_scope": "full_validation_best_checkpoint",
        "split": "val",
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "classification": {
            "source": "ground_truth_future_target_mask_only",
            "mask_channel": FIRE_MASK_CHANNEL,
            "fire_pixel_rule": f"target_mask > {FIRE_MASK_THRESHOLD}",
            "active_fraction_threshold": ACTIVE_FRACTION_THRESHOLD,
            "prediction_threshold": PREDICTED_FIRE_THRESHOLD,
        },
        "dataset_sample_count": expected_samples,
        "evaluated_sample_count": evaluated,
        "unique_sample_id_count": len(sample_ids) if sample_id_count else None,
        "canonical_balance_counts": dict(expected_counts) if expected_counts is not None else None,
        "metrics": metrics,
    }
    per_fire_file_payload = {
        "schema_version": 1,
        "created_at": created_at,
        "metric_scope": "full_validation_per_fire_best_checkpoint",
        "fires": per_fire_payload,
    }
    _atomic_json(run_path / "full_validation_metrics.json", full_payload)
    _atomic_json(run_path / "full_validation_per_fire.json", per_fire_file_payload)
    return {
        "metrics": metrics,
        "per_fire": per_fire_payload,
        "artifact": full_payload,
    }


__all__ = ["FullValidationAccumulator", "evaluate_full_validation"]
