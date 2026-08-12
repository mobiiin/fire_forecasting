"""Evaluate masks derived from multitask consumed-fuel predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
    torch = None

from scripts.diagnose_mask_generalization import (
    EPS,
    _all_mask_definitions_from_raw_sample,
    _build_checkpoint_path,
    _build_datasets,
    _build_loader,
    _ensure_config_path,
    _extract_first_item,
    _resolve_multitask_config,
    _resolve_normalization_reference,
    _save_csv,
    _segmentation_counts,
    _select_plot_ordinals,
    _to_numpy,
)
from src.config import load_config
from src.data.spatial_transforms import infer_with_external_test_spatial_handling
from src.models.convlstm_unet import build_model_from_config
from src.training.checkpoints import load_checkpoint
from src.training.input_normalization import apply_input_normalization, build_input_normalizer_for_loader
from src.training.train import _get_device

CONSUMED_THRESHOLDS = [0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1]


def _metadata_to_dict(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Convert collated metadata to simple Python values."""

    return {key: _extract_first_item(value) for key, value in metadata.items()}


def _denormalize_predicted_consumed_channels(dataset, y_pred: torch.Tensor) -> torch.Tensor:
    """Undo multitask target normalization for the two regression heads when enabled."""

    if not bool(getattr(dataset, "normalize_target", False)):
        return y_pred
    target_mean = getattr(dataset, "target_mean", None)
    target_std = getattr(dataset, "target_std", None)
    if target_mean is None or target_std is None:
        return y_pred
    mean_tensor = torch.as_tensor(target_mean, dtype=y_pred.dtype, device=y_pred.device).reshape(1, -1, 1, 1)
    std_tensor = torch.as_tensor(target_std, dtype=y_pred.dtype, device=y_pred.device).reshape(1, -1, 1, 1)
    std_tensor = torch.clamp(std_tensor, min=1e-6)
    y_pred = y_pred.clone()
    regression_channels = min(int(mean_tensor.shape[1]), 2)
    y_pred[:, :regression_channels] = y_pred[:, :regression_channels] * std_tensor[:, :regression_channels] + mean_tensor[:, :regression_channels]
    return y_pred


def _threshold_metrics_row(
    split_name: str,
    target_mask_name: str,
    consumed_threshold: float,
    counts: Mapping[str, float],
    total_pixels: int,
    sample_count: int,
) -> dict[str, Any]:
    """Convert accumulated counts at one threshold into one CSV row."""

    true_positive = float(counts["tp"])
    false_positive = float(counts["fp"])
    false_negative = float(counts["fn"])
    iou = true_positive / (true_positive + false_positive + false_negative + EPS)
    dice = (2.0 * true_positive) / (2.0 * true_positive + false_positive + false_negative + EPS)
    precision = true_positive / (true_positive + false_positive + EPS)
    recall = true_positive / (true_positive + false_negative + EPS)
    return {
        "split": split_name,
        "target_mask": target_mask_name,
        "consumed_threshold": float(consumed_threshold),
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "predicted_active_fraction": float(float(counts["pred_active_pixels"]) / max(total_pixels, 1)),
        "true_active_fraction": float(float(counts["true_active_pixels"]) / max(total_pixels, 1)),
        "samples_with_predicted_active": int(counts["samples_with_pred"]),
        "samples_with_true_active": int(counts["samples_with_true"]),
        "sample_count": int(sample_count),
        "pixel_count": int(total_pixels),
    }


def _save_derived_mask_plot(
    split_name: str,
    output_path: Path,
    sample_index: int,
    pred_surface_consumed: np.ndarray,
    pred_canopy_consumed: np.ndarray,
    pred_consumed_max: np.ndarray,
    derived_pred_mask: np.ndarray,
    active_flux_mask: np.ndarray,
    burned_fuel_mask: np.ndarray,
    representative_threshold: float,
) -> None:
    """Save one derived-mask comparison figure."""

    panels = [
        ("Pred surface consumed", pred_surface_consumed, "inferno"),
        ("Pred canopy consumed", pred_canopy_consumed, "inferno"),
        ("Pred consumed max", pred_consumed_max, "magma"),
        (f"Derived pred mask @ {representative_threshold:g}", derived_pred_mask, "viridis"),
        ("True active-flux mask", active_flux_mask, "viridis"),
        ("True burned-fuel mask", burned_fuel_mask, "viridis"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150, constrained_layout=True)
    for axis, (title, panel_data, cmap) in zip(axes.flatten(), panels, strict=True):
        panel_data = np.asarray(panel_data, dtype=np.float32)
        if "mask" in title.lower():
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.nanmin(panel_data))
            vmax = float(np.nanmax(panel_data))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0
        image = axis.imshow(panel_data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"{split_name} sample_index={sample_index} | derived consumed-fuel mask", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_split(
    split_name: str,
    dataset,
    model,
    config: Mapping[str, Any],
    device,
    num_visualizations: int,
    output_root: Path,
) -> list[dict[str, Any]]:
    """Evaluate consumed-fuel-derived masks on one split."""

    loader = _build_loader(dataset)
    input_channels = int(getattr(dataset, "total_input_channels", getattr(dataset, "input_channel_count", 0)))
    input_normalizer = build_input_normalizer_for_loader(loader, device, input_channels, config) if input_channels > 0 else None
    plot_ordinals = _select_plot_ordinals(len(dataset), num_visualizations)
    representative_threshold = float(_resolve_multitask_config(config)["consumed_fuel_threshold"])

    accumulators = {
        target_mask_name: {
            threshold: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "pred_active_pixels": 0, "true_active_pixels": 0, "samples_with_pred": 0, "samples_with_true": 0}
            for threshold in CONSUMED_THRESHOLDS
        }
        for target_mask_name in ("active_flux", "burned_fuel")
    }
    total_pixels = 0

    model.eval()
    with torch.no_grad():
        for ordinal, batch in enumerate(loader):
            if not isinstance(batch, (tuple, list)) or len(batch) < 3:
                raise TypeError("Expected batches to contain input tensor, target tensor, and metadata.")
            x_batch = batch[0].to(device)
            x_batch = apply_input_normalization(x_batch, input_normalizer, config)
            metadata = _metadata_to_dict(batch[2])

            if split_name == "external_test":
                inference_result = infer_with_external_test_spatial_handling(model, x_batch, config)
                y_pred = inference_result["y_pred"]
            else:
                y_pred = model(x_batch)
            y_pred = _denormalize_predicted_consumed_channels(dataset, y_pred)

            pred_surface_consumed = _to_numpy(y_pred[:, 0:1])[0, 0]
            pred_canopy_consumed = _to_numpy(y_pred[:, 1:2])[0, 0]
            pred_consumed_max = np.maximum(pred_surface_consumed, pred_canopy_consumed).astype(np.float32, copy=False)
            total_pixels += int(pred_consumed_max.size)

            definitions = _all_mask_definitions_from_raw_sample(dataset, metadata, config)
            active_flux_mask = np.asarray(definitions["active_flux"]["mask"], dtype=np.float32)
            burned_fuel_mask = np.asarray(definitions["burned_fuel"]["mask"], dtype=np.float32)

            for threshold in CONSUMED_THRESHOLDS:
                derived_pred_mask = (pred_consumed_max > threshold).astype(np.float32, copy=False)
                for target_mask_name, true_mask in (
                    ("active_flux", active_flux_mask),
                    ("burned_fuel", burned_fuel_mask),
                ):
                    true_positive, false_positive, false_negative = _segmentation_counts(derived_pred_mask, true_mask)
                    accumulator = accumulators[target_mask_name][threshold]
                    accumulator["tp"] += true_positive
                    accumulator["fp"] += false_positive
                    accumulator["fn"] += false_negative
                    accumulator["pred_active_pixels"] += int(np.sum(derived_pred_mask))
                    accumulator["true_active_pixels"] += int(np.sum(true_mask))
                    accumulator["samples_with_pred"] += int(np.any(derived_pred_mask > 0.5))
                    accumulator["samples_with_true"] += int(np.any(true_mask > 0.5))

            if ordinal in plot_ordinals:
                derived_pred_mask = (pred_consumed_max > representative_threshold).astype(np.float32, copy=False)
                sample_index = int(metadata.get("sample_index", ordinal))
                output_path = output_root / split_name / f"sample_{sample_index:05d}.png"
                _save_derived_mask_plot(
                    split_name=split_name,
                    output_path=output_path,
                    sample_index=sample_index,
                    pred_surface_consumed=pred_surface_consumed,
                    pred_canopy_consumed=pred_canopy_consumed,
                    pred_consumed_max=pred_consumed_max,
                    derived_pred_mask=derived_pred_mask,
                    active_flux_mask=active_flux_mask,
                    burned_fuel_mask=burned_fuel_mask,
                    representative_threshold=representative_threshold,
                )

    sample_count = int(len(dataset))
    rows: list[dict[str, Any]] = []
    for target_mask_name, threshold_map in accumulators.items():
        for threshold, counts in threshold_map.items():
            rows.append(_threshold_metrics_row(split_name, target_mask_name, threshold, counts, total_pixels, sample_count))
    return rows


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Evaluate masks derived from predicted consumed fuel.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path.")
    parser.add_argument(
        "--checkpoint-kind",
        choices=("best", "latest"),
        default="best",
        help="Which config-derived checkpoint to use when --checkpoint is omitted.",
    )
    parser.add_argument("--max_samples", type=int, default=200, help="Maximum chronological samples per split.")
    parser.add_argument("--num_visualizations", type=int, default=6, help="Maximum figures to save per split.")
    return parser


def main() -> None:
    """CLI entry point."""

    if torch is None:
        raise ImportError("PyTorch is required for evaluate_consumed_fuel_derived_mask.py.")

    args = build_argument_parser().parse_args()
    config = _ensure_config_path(load_config(args.config), args.config)
    if str(config.get("task_type", "regression")).lower() != "multitask":
        raise ValueError("evaluate_consumed_fuel_derived_mask.py requires task_type: multitask.")
    if config.get("test_data_dir") in (None, "", "null"):
        raise ValueError("evaluate_consumed_fuel_derived_mask.py requires test_data_dir for the external test split.")

    datasets = _build_datasets(config, _resolve_normalization_reference(config), int(args.max_samples))
    input_channels = int(datasets["train"].total_input_channels)
    device = _get_device(config)
    model = build_model_from_config(config, input_channels=input_channels).to(device)

    checkpoint_path = _build_checkpoint_path(config, args.checkpoint, args.checkpoint_kind)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    output_root = Path("outputs/consumed_fuel_derived_mask").expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for split_name, dataset in datasets.items():
        rows.extend(evaluate_split(split_name, dataset, model, config, device, int(args.num_visualizations), output_root))

    csv_path = Path("artifacts/logs/consumed_fuel_derived_mask_metrics.csv").expanduser().resolve()
    _save_csv(rows, csv_path)

    print("Consumed-Fuel Derived Mask Evaluation")
    print("Derived prediction mask = max(pred_surface_consumed, pred_canopy_consumed) > consumed_threshold")
    print(f"consumed_threshold sweep: {CONSUMED_THRESHOLDS}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figures: {output_root}")


if __name__ == "__main__":
    main()
