"""Diagnose multitask mask generalization across train, validation, and external test splits."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
    torch = None
    DataLoader = None

from scripts.evaluate_persistence_baseline import (
    crop_channel_map,
    discover_external_test_files,
    discover_files,
)
from src.config import load_config
from src.data.dataset import FireSequenceDataset, _resolve_multitask_config
from src.data.spatial_transforms import infer_with_external_test_spatial_handling
from src.data.splits import chronological_split_indices, chronological_train_val_split_indices
from src.models.convlstm_unet import build_model_from_config
from src.training.checkpoints import latest_and_best_checkpoint_paths, load_checkpoint
from src.training.train import _ensure_config_path, _get_device

SOURCE_THRESHOLDS = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
PREDICTION_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
FINE_PREDICTION_THRESHOLDS = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.3, 0.5]
PERCENTILES = [50.0, 75.0, 90.0, 95.0, 99.0, 99.5, 99.9]
EPS = 1.0e-6


def _get_section(config: Mapping[str, Any], *names: str) -> dict[str, Any]:
    """Return the first mapping found under any of the requested section names."""

    for name in names:
        section = config.get(name)
        if isinstance(section, Mapping):
            return dict(section)
    return {}


def _resolve_path(base_path: Path | None, configured_path: str | Path) -> Path:
    """Resolve a configured path relative to the config location when needed."""

    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base_path is None:
        return path.resolve()
    return (base_path.parent / path).resolve()


def _extract_first_item(value):
    """Convert collated batch metadata values into a single scalar or string."""

    if torch is not None and torch.is_tensor(value):
        return value.reshape(-1)[0].item() if value.numel() else None
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _metadata_to_dict(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a collated metadata mapping into plain Python scalars."""

    return {key: _extract_first_item(value) for key, value in metadata.items()}


def _select_evenly_spaced(values: Sequence[int], maximum_items: int) -> list[int]:
    """Return at most ``maximum_items`` chronological entries sampled evenly."""

    concrete = [int(value) for value in values]
    if maximum_items <= 0 or not concrete:
        return []
    if len(concrete) <= maximum_items:
        return concrete
    raw_positions = np.linspace(0, len(concrete) - 1, num=maximum_items)
    selected_positions = sorted({int(round(position)) for position in raw_positions})
    return [concrete[position] for position in selected_positions]


def _select_plot_ordinals(sample_count: int, maximum_plots: int) -> set[int]:
    """Choose chronological sample ordinals to visualize."""

    if sample_count <= 0 or maximum_plots <= 0:
        return set()
    if sample_count <= maximum_plots:
        return set(range(sample_count))
    raw_positions = np.linspace(0, sample_count - 1, num=maximum_plots)
    return {int(round(position)) for position in raw_positions}


def _to_numpy(array_like) -> np.ndarray:
    """Convert a tensor-like object to a float32 NumPy array."""

    if torch is not None and torch.is_tensor(array_like):
        return array_like.detach().cpu().to(torch.float32).numpy()
    return np.asarray(array_like, dtype=np.float32)


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    """Compute descriptive statistics over finite values."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _percentile_stats(values: np.ndarray) -> dict[str, float]:
    """Compute the configured percentile set over finite values."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"p{percentile:g}": math.nan for percentile in PERCENTILES}
    percentile_values = np.percentile(finite, PERCENTILES)
    return {
        f"p{percentile:g}": float(value)
        for percentile, value in zip(PERCENTILES, percentile_values, strict=True)
    }


def _format_threshold_label(value: float) -> str:
    """Build a stable threshold label suitable for CSV column names."""

    return f"{value:g}".replace("-", "neg").replace(".", "_")


def _save_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write a list of dictionary rows to a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sigmoid(probabilities_input: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid wrapper."""

    if torch is None:
        raise ImportError("PyTorch is required for mask diagnostics.")
    return torch.sigmoid(probabilities_input)


def _build_checkpoint_path(config: Mapping[str, Any], checkpoint_path: str | None, checkpoint_kind: str) -> Path:
    """Resolve the checkpoint path, preferring the config-derived best checkpoint."""

    if checkpoint_path:
        return Path(checkpoint_path).expanduser().resolve()
    checkpoint_config = _get_section(config, "checkpoint")
    configured_checkpoint = checkpoint_config.get("path", "./artifacts/checkpoints/convlstm_unet.pt")
    config_path_value = config.get("config_path", config.get("_config_path"))
    config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
    latest_path, best_path = latest_and_best_checkpoint_paths(_resolve_path(config_path, configured_checkpoint))
    return best_path if checkpoint_kind == "best" else latest_path


def _resolve_normalization_reference(config: Mapping[str, Any]) -> str | Path | None:
    """Return the configured normalization stats path when it exists."""

    normalization_config = _get_section(config, "normalization")
    configured_path = normalization_config.get("path")
    if not configured_path:
        return None
    config_path_value = config.get("config_path", config.get("_config_path"))
    config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
    resolved = _resolve_path(config_path, configured_path)
    return resolved if resolved.exists() else None


def _build_datasets(
    config: Mapping[str, Any],
    normalization_reference: str | Path | None,
    max_samples: int,
) -> dict[str, FireSequenceDataset]:
    """Build chronological train/val/external-test datasets with optional subsampling."""

    input_sequence_length = int(config["input_sequence_length"])
    prediction_horizon = int(config["prediction_horizon"])
    task_type = str(config.get("task_type", "regression")).lower()
    if task_type != "multitask":
        raise ValueError(
            "diagnose_mask_generalization.py only supports task_type='multitask'. "
            f"Got {task_type!r}."
        )

    common_kwargs = {
        "input_sequence_length": input_sequence_length,
        "prediction_horizon": prediction_horizon,
        "target_channel": int(config.get("target_channel", 0)),
        "input_channel_count": int(config.get("input_channel_count", _get_section(config, "model").get("input_channels", 0))),
        "input_channel_indices": config.get("input_channel_indices"),
        "task_type": task_type,
        "fire_threshold": float(config.get("fire_threshold", 0.5)),
        "use_patches": False,
        "normalization_stats": normalization_reference,
        "normalize_target": bool(_get_section(config, "target_normalization").get("enabled", False)),
        "return_metadata": True,
        "config": config,
    }

    main_files = discover_files(Path(str(config["config_path"])), config)
    split_mode = str(config.get("split_mode", "train_val_test")).lower()
    if split_mode == "train_val_external_test":
        splits = chronological_train_val_split_indices(
            num_timesteps=len(main_files),
            input_sequence_length=input_sequence_length,
            prediction_horizon=prediction_horizon,
            train_fraction=float(config.get("train_fraction", 0.85)),
            val_fraction=float(config.get("val_fraction", 0.15)),
        )
    else:
        splits = chronological_split_indices(
            num_timesteps=len(main_files),
            input_sequence_length=input_sequence_length,
            prediction_horizon=prediction_horizon,
            train_fraction=float(config.get("train_fraction", 0.7)),
            val_fraction=float(config.get("val_fraction", 0.15)),
            test_fraction=float(config.get("test_fraction", 0.15)),
            split_mode=split_mode,
        )

    train_indices = _select_evenly_spaced(splits["train"], max_samples)
    val_indices = _select_evenly_spaced(splits["val"], max_samples)
    external_files = discover_external_test_files(Path(str(config["config_path"])), config)
    external_dataset = FireSequenceDataset(
        file_paths=external_files,
        sample_indices=_select_evenly_spaced(
            list(range(max(0, len(external_files) - input_sequence_length - prediction_horizon + 1))),
            max_samples,
        ),
        **common_kwargs,
    )

    return {
        "train": FireSequenceDataset(file_paths=main_files, sample_indices=train_indices, **common_kwargs),
        "val": FireSequenceDataset(file_paths=main_files, sample_indices=val_indices, **common_kwargs),
        "external_test": external_dataset,
    }


def _build_loader(dataset: FireSequenceDataset) -> Any:
    """Create a deterministic batch-size-1 loader for a diagnostic dataset."""

    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required to build diagnostic DataLoaders.")
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _crop_with_metadata(channel_map: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    """Crop a 2D map to the evaluated patch when metadata contains patch coordinates."""

    return crop_channel_map(
        channel_map,
        patch_top=metadata.get("patch_top"),
        patch_left=metadata.get("patch_left"),
        patch_size=metadata.get("patch_size"),
    )


def _all_mask_definitions_from_raw_sample(
    dataset: FireSequenceDataset,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build both active-flux and burned-fuel mask definitions from raw files for one sample."""

    multitask = _resolve_multitask_config(config)
    future_frame = np.asarray(
        np.load(Path(str(metadata["target_file_path"])).expanduser().resolve(), mmap_mode="r", allow_pickle=False),
        dtype=np.float32,
    )
    if dataset.initial_fuel_map is None:
        raise ValueError("Dataset initial_fuel_map is required for multitask mask diagnostics.")

    flux_channel = int(multitask["flux_mask_channel"])
    future_flux = np.asarray(future_frame[:, :, flux_channel], dtype=np.float32)
    active_flux_threshold = float(multitask["flux_fire_threshold"])
    active_flux_mask = (future_flux > active_flux_threshold).astype(np.float32, copy=False)

    surface_channel = int(multitask["surface_fuel_channel"])
    canopy_channel = int(multitask["canopy_fuel_channel"])
    future_surface_fuel = np.asarray(future_frame[:, :, surface_channel], dtype=np.float32)
    future_canopy_fuel = np.asarray(future_frame[:, :, canopy_channel], dtype=np.float32)
    initial_surface_fuel = np.asarray(dataset.initial_fuel_map[:, :, 0], dtype=np.float32)
    initial_canopy_fuel = np.asarray(dataset.initial_fuel_map[:, :, 1], dtype=np.float32)
    surface_cumulative_consumed = initial_surface_fuel - future_surface_fuel
    canopy_cumulative_consumed = initial_canopy_fuel - future_canopy_fuel
    if bool(multitask["clamp_consumed_fuel_targets_nonnegative"]):
        surface_cumulative_consumed = np.maximum(surface_cumulative_consumed, 0.0)
        canopy_cumulative_consumed = np.maximum(canopy_cumulative_consumed, 0.0)
    combined_cumulative_consumed = np.maximum(surface_cumulative_consumed, canopy_cumulative_consumed).astype(np.float32, copy=False)
    burned_fuel_threshold = float(multitask["consumed_fuel_threshold"])
    burned_fuel_mask = (combined_cumulative_consumed > burned_fuel_threshold).astype(np.float32, copy=False)

    return {
        "active_flux": {
            "source_map": _crop_with_metadata(future_flux, metadata),
            "mask": _crop_with_metadata(active_flux_mask, metadata),
            "source_label": f"future_flux_channel_{flux_channel}",
            "threshold": active_flux_threshold,
        },
        "burned_fuel": {
            "source_map": _crop_with_metadata(combined_cumulative_consumed, metadata),
            "mask": _crop_with_metadata(burned_fuel_mask, metadata),
            "source_label": "combined_cumulative_consumed_fuel",
            "threshold": burned_fuel_threshold,
        },
    }


def _mask_source_from_raw_sample(
    dataset: FireSequenceDataset,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """Rebuild the raw source map and true binary mask for one multitask sample."""

    multitask = _resolve_multitask_config(config)
    mask_target_type = str(multitask["mask_target_type"]).lower()
    all_definitions = _all_mask_definitions_from_raw_sample(dataset, metadata, config)
    if mask_target_type not in all_definitions:
        raise ValueError(
            "Unsupported multitask.mask_target_type. "
            f"Expected 'active_flux' or 'burned_fuel', got {mask_target_type!r}."
        )
    selected = all_definitions[mask_target_type]

    return (
        np.asarray(selected["source_map"], dtype=np.float32),
        np.asarray(selected["mask"], dtype=np.float32),
        str(selected["source_label"]),
        float(selected["threshold"]),
    )


def _source_definition(config: Mapping[str, Any]) -> str:
    """Return a human-readable description of the configured mask target."""

    multitask = _resolve_multitask_config(config)
    mask_target_type = str(multitask["mask_target_type"]).lower()
    if mask_target_type == "active_flux":
        return (
            "mask = future_frame[:, :, flux_mask_channel] > flux_fire_threshold "
            f"(channel={int(multitask['flux_mask_channel'])}, threshold={float(multitask['flux_fire_threshold']):.6g})"
        )
    return (
        "mask = max(initial_surface_fuel - future_surface_fuel, "
        "initial_canopy_fuel - future_canopy_fuel) > consumed_fuel_threshold "
        f"(threshold={float(multitask['consumed_fuel_threshold']):.6g})"
    )


def _segmentation_counts(predicted_mask: np.ndarray, true_mask: np.ndarray) -> tuple[float, float, float]:
    """Return TP/FP/FN counts for float binary masks."""

    predicted_mask = np.asarray(predicted_mask, dtype=np.float32)
    true_mask = np.asarray(true_mask, dtype=np.float32)
    true_positive = float(np.sum(predicted_mask * true_mask))
    false_positive = float(np.sum(predicted_mask * (1.0 - true_mask)))
    false_negative = float(np.sum((1.0 - predicted_mask) * true_mask))
    return true_positive, false_positive, false_negative


def _threshold_metrics_row(
    split_name: str,
    threshold: float,
    counts: Mapping[str, float],
    total_pixels: int,
    sample_count: int,
) -> dict[str, Any]:
    """Convert accumulated counts at one threshold into a metric row."""

    true_positive = float(counts["tp"])
    false_positive = float(counts["fp"])
    false_negative = float(counts["fn"])
    iou = true_positive / (true_positive + false_positive + false_negative + EPS)
    dice = (2.0 * true_positive) / (2.0 * true_positive + false_positive + false_negative + EPS)
    precision = true_positive / (true_positive + false_positive + EPS)
    recall = true_positive / (true_positive + false_negative + EPS)
    return {
        "split": split_name,
        "prediction_threshold": float(threshold),
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "active_fraction_true": float(float(counts["true_active_pixels"]) / max(total_pixels, 1)),
        "active_fraction_predicted": float(float(counts["pred_active_pixels"]) / max(total_pixels, 1)),
        "samples_with_true_active": int(counts["samples_with_true"]),
        "samples_with_predicted_active": int(counts["samples_with_pred"]),
        "sample_count": int(sample_count),
        "pixel_count": int(total_pixels),
    }


def _save_mask_diagnostic_plot(
    split_name: str,
    output_path: Path,
    title: str,
    source_label: str,
    source_map: np.ndarray,
    true_mask: np.ndarray,
    probability_map: np.ndarray,
) -> None:
    """Save a six-panel diagnostic figure for one mask sample."""

    source_map = np.asarray(source_map, dtype=np.float32)
    true_mask = np.asarray(true_mask, dtype=np.float32)
    probability_map = np.asarray(probability_map, dtype=np.float32)
    predicted_mask_05 = (probability_map >= 0.5).astype(np.float32)
    predicted_mask_02 = (probability_map >= 0.2).astype(np.float32)

    source_vmin = float(np.nanmin(source_map))
    source_vmax = float(np.nanmax(source_map))
    if np.isclose(source_vmin, source_vmax):
        source_vmax = source_vmin + 1.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150, constrained_layout=True)
    panels = [
        (f"{source_label}", source_map, "inferno", source_vmin, source_vmax),
        ("True mask", true_mask, "viridis", 0.0, 1.0),
        ("Predicted mask probability", probability_map, "magma", 0.0, 1.0),
        ("Predicted mask @ 0.5", predicted_mask_05, "viridis", 0.0, 1.0),
        ("Predicted mask @ 0.2", predicted_mask_02, "viridis", 0.0, 1.0),
        ("Contour overlay (true cyan, pred@0.5 white)", source_map, "inferno", source_vmin, source_vmax),
    ]

    for axis, (panel_title, panel_data, cmap, vmin, vmax) in zip(axes.flatten(), panels, strict=True):
        image = axis.imshow(panel_data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(panel_title)
        axis.set_xticks([])
        axis.set_yticks([])
        if "Contour overlay" in panel_title:
            if np.nanmin(true_mask) <= 0.5 <= np.nanmax(true_mask):
                axis.contour(true_mask, levels=[0.5], colors=["cyan"], linewidths=1.8)
            if np.nanmin(predicted_mask_05) <= 0.5 <= np.nanmax(predicted_mask_05):
                axis.contour(predicted_mask_05, levels=[0.5], colors=["white"], linewidths=1.8, linestyles=["--"])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(f"{title}\n{split_name}", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def diagnose_split(
    split_name: str,
    dataset: FireSequenceDataset,
    model,
    config: Mapping[str, Any],
    device: torch.device,
    num_visualizations: int,
    visualization_root: Path,
) -> dict[str, Any]:
    """Run raw-source, probability, threshold-sweep, and plotting diagnostics for one split."""

    loader = _build_loader(dataset)
    plot_ordinals = _select_plot_ordinals(len(dataset), num_visualizations) if split_name in {"val", "external_test"} else set()

    source_values: list[np.ndarray] = []
    logit_values: list[np.ndarray] = []
    probability_values: list[np.ndarray] = []
    source_any_counts = {threshold: 0 for threshold in SOURCE_THRESHOLDS}
    source_active_pixels = {threshold: 0 for threshold in SOURCE_THRESHOLDS}
    pred_any_counts = {threshold: 0 for threshold in PREDICTION_THRESHOLDS}
    pred_active_pixels = {threshold: 0 for threshold in PREDICTION_THRESHOLDS}
    segmentation_accumulators = {
        threshold: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "pred_active_pixels": 0, "true_active_pixels": 0, "samples_with_pred": 0, "samples_with_true": 0}
        for threshold in PREDICTION_THRESHOLDS
    }
    fine_segmentation_accumulators = {
        threshold: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "pred_active_pixels": 0, "true_active_pixels": 0, "samples_with_pred": 0, "samples_with_true": 0}
        for threshold in FINE_PREDICTION_THRESHOLDS
    }
    mode_counts: dict[str, int] = {}
    source_label = "mask_source"
    configured_source_threshold = math.nan
    total_pixels = 0

    model.eval()
    with torch.no_grad():
        for ordinal, batch in enumerate(loader):
            if not isinstance(batch, (tuple, list)) or len(batch) < 3:
                raise TypeError("Expected diagnostic batches to contain input tensor, target tensor, and metadata.")
            x_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            metadata = _metadata_to_dict(batch[2])
            mode_key = "direct"

            if split_name == "external_test":
                inference_result = infer_with_external_test_spatial_handling(model, x_batch, config)
                y_pred = inference_result["y_pred"]
                mode_key = str(inference_result["mode_used"])
                mode_counts[mode_key] = mode_counts.get(mode_key, 0) + 1
            else:
                y_pred = model(x_batch)
                mode_counts["direct"] = mode_counts.get("direct", 0) + 1

            if tuple(y_pred.shape[-2:]) != tuple(y_batch.shape[-2:]):
                raise ValueError(
                    f"{split_name} prediction/target spatial mismatch after inference: "
                    f"pred={tuple(y_pred.shape)} target={tuple(y_batch.shape)}."
                )

            true_mask = _to_numpy(y_batch[:, 2:3])
            logits = _to_numpy(y_pred[:, 2:3])
            probability = _to_numpy(_sigmoid(y_pred[:, 2:3]))
            source_map, raw_true_mask, source_label, configured_source_threshold = _mask_source_from_raw_sample(dataset, metadata, config)

            true_mask_2d = np.asarray(true_mask[0, 0], dtype=np.float32)
            logits_2d = np.asarray(logits[0, 0], dtype=np.float32)
            probability_2d = np.asarray(probability[0, 0], dtype=np.float32)
            raw_true_mask = np.asarray(raw_true_mask, dtype=np.float32)
            if not np.allclose(true_mask_2d, raw_true_mask, atol=1e-6):
                raise ValueError(
                    f"{split_name} mask target mismatch between dataset y[:,2] and raw reconstructed mask "
                    f"for sample_index={metadata.get('sample_index')}."
                )

            source_values.append(source_map.reshape(-1))
            logit_values.append(logits_2d.reshape(-1))
            probability_values.append(probability_2d.reshape(-1))
            total_pixels += int(source_map.size)

            for threshold in SOURCE_THRESHOLDS:
                active_pixels = source_map > threshold
                source_any_counts[threshold] += int(np.any(active_pixels))
                source_active_pixels[threshold] += int(np.sum(active_pixels))

            for threshold in PREDICTION_THRESHOLDS:
                predicted_mask = (probability_2d >= threshold).astype(np.float32)
                pred_any_counts[threshold] += int(np.any(predicted_mask > 0.5))
                pred_active_pixels[threshold] += int(np.sum(predicted_mask))
                true_positive, false_positive, false_negative = _segmentation_counts(predicted_mask, true_mask_2d)
                accumulator = segmentation_accumulators[threshold]
                accumulator["tp"] += true_positive
                accumulator["fp"] += false_positive
                accumulator["fn"] += false_negative
                accumulator["pred_active_pixels"] += int(np.sum(predicted_mask))
                accumulator["true_active_pixels"] += int(np.sum(true_mask_2d))
                accumulator["samples_with_pred"] += int(np.any(predicted_mask > 0.5))
                accumulator["samples_with_true"] += int(np.any(true_mask_2d > 0.5))

            for threshold in FINE_PREDICTION_THRESHOLDS:
                predicted_mask = (probability_2d >= threshold).astype(np.float32)
                true_positive, false_positive, false_negative = _segmentation_counts(predicted_mask, true_mask_2d)
                accumulator = fine_segmentation_accumulators[threshold]
                accumulator["tp"] += true_positive
                accumulator["fp"] += false_positive
                accumulator["fn"] += false_negative
                accumulator["pred_active_pixels"] += int(np.sum(predicted_mask))
                accumulator["true_active_pixels"] += int(np.sum(true_mask_2d))
                accumulator["samples_with_pred"] += int(np.any(predicted_mask > 0.5))
                accumulator["samples_with_true"] += int(np.any(true_mask_2d > 0.5))

            if ordinal in plot_ordinals:
                sample_index = int(metadata.get("sample_index", ordinal))
                output_path = visualization_root / split_name / f"sample_{sample_index:05d}.png"
                _save_mask_diagnostic_plot(
                    split_name=split_name,
                    output_path=output_path,
                    title=(
                        f"sample_index={sample_index} | output channel 2 = mask logits | "
                        f"native grid {tuple(true_mask_2d.shape)} | inference_mode={mode_key}"
                    ),
                    source_label=source_label,
                    source_map=source_map,
                    true_mask=true_mask_2d,
                    probability_map=probability_2d,
                )

    if not source_values:
        raise ValueError(f"{split_name} dataset produced no samples for mask diagnostics.")

    source_array = np.concatenate(source_values).astype(np.float32, copy=False)
    logits_array = np.concatenate(logit_values).astype(np.float32, copy=False)
    probability_array = np.concatenate(probability_values).astype(np.float32, copy=False)
    sample_count = int(len(dataset))

    source_summary = {
        "split": split_name,
        "sample_count": sample_count,
        "pixel_count": total_pixels,
        "source_label": source_label,
        "configured_mask_threshold": float(configured_source_threshold),
    }
    source_summary.update(_finite_stats(source_array))
    source_summary.update(_percentile_stats(source_array))
    for threshold in SOURCE_THRESHOLDS:
        key = _format_threshold_label(threshold)
        source_summary[f"active_pixel_fraction_at_{key}"] = float(source_active_pixels[threshold] / max(total_pixels, 1))
        source_summary[f"samples_with_any_active_at_{key}"] = int(source_any_counts[threshold])
        source_summary[f"sample_fraction_with_any_active_at_{key}"] = float(source_any_counts[threshold] / max(sample_count, 1))

    probability_summary = {
        "split": split_name,
        "sample_count": sample_count,
        "pixel_count": total_pixels,
        "mode_counts": ",".join(f"{mode}:{count}" for mode, count in sorted(mode_counts.items())),
    }
    probability_summary.update({f"logits_{key}": value for key, value in _finite_stats(logits_array).items()})
    probability_summary.update({f"prob_{key}": value for key, value in _finite_stats(probability_array).items()})
    probability_summary.update({f"prob_{key}": value for key, value in _percentile_stats(probability_array).items()})
    for threshold in PREDICTION_THRESHOLDS:
        key = _format_threshold_label(threshold)
        probability_summary[f"predicted_active_pixel_fraction_at_{key}"] = float(pred_active_pixels[threshold] / max(total_pixels, 1))
        probability_summary[f"samples_with_any_prediction_at_{key}"] = int(pred_any_counts[threshold])
        probability_summary[f"sample_fraction_with_any_prediction_at_{key}"] = float(pred_any_counts[threshold] / max(sample_count, 1))

    threshold_rows: list[dict[str, Any]] = []
    for threshold in PREDICTION_THRESHOLDS:
        threshold_rows.append(_threshold_metrics_row(split_name, threshold, segmentation_accumulators[threshold], total_pixels, sample_count))

    fine_threshold_rows: list[dict[str, Any]] = []
    for threshold in FINE_PREDICTION_THRESHOLDS:
        fine_threshold_rows.append(
            _threshold_metrics_row(split_name, threshold, fine_segmentation_accumulators[threshold], total_pixels, sample_count)
        )

    return {
        "source_summary": source_summary,
        "probability_summary": probability_summary,
        "threshold_rows": threshold_rows,
        "fine_threshold_rows": fine_threshold_rows,
    }


def _find_threshold_row(rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    """Return the metrics row for one prediction threshold when present."""

    for row in rows:
        if math.isclose(float(row["prediction_threshold"]), float(threshold), rel_tol=0.0, abs_tol=1e-9):
            return row
    return None


def _build_warnings(
    config: Mapping[str, Any],
    source_rows: dict[str, dict[str, Any]],
    probability_rows: dict[str, dict[str, Any]],
    threshold_rows_by_split: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Generate high-signal warnings from the collected diagnostics."""

    warnings: list[str] = []
    multitask = _resolve_multitask_config(config)
    mask_target_type = str(multitask["mask_target_type"]).lower()
    external_source = source_rows["external_test"]
    external_probability = probability_rows["external_test"]
    train_source = source_rows["train"]
    val_source = source_rows["val"]

    configured_source_threshold = float(external_source["configured_mask_threshold"])
    configured_key = _format_threshold_label(configured_source_threshold)
    external_active_fraction = float(external_source.get(f"active_pixel_fraction_at_{configured_key}", 0.0))
    train_active_fraction = float(train_source.get(f"active_pixel_fraction_at_{configured_key}", 0.0))
    val_active_fraction = float(val_source.get(f"active_pixel_fraction_at_{configured_key}", 0.0))

    if external_active_fraction <= 1.0e-6 and max(train_active_fraction, val_active_fraction) > 1.0e-5:
        if mask_target_type == "active_flux":
            warnings.append("Configured flux_fire_threshold may be too high for external test dataset.")
        else:
            warnings.append("Configured consumed_fuel_threshold may be too high for external test dataset.")

    external_prob_99 = float(external_probability.get("prob_p99", math.nan))
    pred_frac_05 = float(external_probability.get(f"predicted_active_pixel_fraction_at_{_format_threshold_label(0.5)}", 0.0))
    pred_frac_02 = float(external_probability.get(f"predicted_active_pixel_fraction_at_{_format_threshold_label(0.2)}", 0.0))
    if np.isfinite(external_prob_99) and external_prob_99 >= 0.2 and pred_frac_05 <= 1.0e-6 and pred_frac_02 > pred_frac_05 * 10.0:
        warnings.append("Prediction threshold may be too strict; calibration may be needed.")

    if mask_target_type == "active_flux":
        external_flux_p99 = float(external_source.get("p99", math.nan))
        reference_flux_p99 = max(float(train_source.get("p99", math.nan)), float(val_source.get("p99", math.nan)))
        if np.isfinite(external_flux_p99) and np.isfinite(reference_flux_p99) and reference_flux_p99 > 0.0 and external_flux_p99 < reference_flux_p99 * 0.1:
            warnings.append("External dataset flux scale differs from training dataset.")
        if abs(float(external_source.get("max", 0.0))) < 1.0e-8 and abs(external_flux_p99) < 1.0e-8:
            warnings.append("Check channel layout or units for flux_mask_channel.")

    row_005 = _find_threshold_row(threshold_rows_by_split["external_test"], 0.05)
    if external_active_fraction > 1.0e-5 and np.isfinite(external_prob_99) and external_prob_99 < 0.05:
        if row_005 is None or float(row_005.get("active_fraction_predicted", 0.0)) <= 1.0e-6:
            warnings.append("Likely domain shift or insufficient generalization.")

    return warnings


def _print_summary(
    config: Mapping[str, Any],
    source_rows: dict[str, dict[str, Any]],
    probability_rows: dict[str, dict[str, Any]],
    threshold_rows_by_split: dict[str, list[dict[str, Any]]],
    warnings: Sequence[str],
    csv_paths: Sequence[Path],
    visualization_root: Path,
) -> None:
    """Print a readable terminal summary of the diagnostics."""

    multitask = _resolve_multitask_config(config)
    print("Mask Generalization Diagnostics")
    print("Output channel 2 is treated as segmentation mask logits, not as raw flux regression.")
    print(f"Mask target definition: {_source_definition(config)}")
    print(f"mask_target_type: {multitask['mask_target_type']}")
    print(f"source thresholds swept: {SOURCE_THRESHOLDS}")
    print(f"prediction thresholds swept: {PREDICTION_THRESHOLDS}")
    print(f"fine prediction thresholds swept: {FINE_PREDICTION_THRESHOLDS}")
    print("")

    for split_name in ("train", "val", "external_test"):
        source_row = source_rows[split_name]
        probability_row = probability_rows[split_name]
        metrics_05 = _find_threshold_row(threshold_rows_by_split[split_name], 0.5)
        configured_threshold = float(source_row["configured_mask_threshold"])
        configured_key = _format_threshold_label(configured_threshold)
        print(f"[{split_name}]")
        print(
            f"  raw_source: min={float(source_row['min']):.6g} max={float(source_row['max']):.6g} "
            f"mean={float(source_row['mean']):.6g} std={float(source_row['std']):.6g} p99={float(source_row['p99']):.6g}"
        )
        print(
            f"  configured_source_active_fraction: "
            f"{float(source_row.get(f'active_pixel_fraction_at_{configured_key}', 0.0)):.6g}"
        )
        print(
            f"  mask_logits: min={float(probability_row['logits_min']):.6g} max={float(probability_row['logits_max']):.6g} "
            f"mean={float(probability_row['logits_mean']):.6g} std={float(probability_row['logits_std']):.6g}"
        )
        print(
            f"  mask_probabilities: min={float(probability_row['prob_min']):.6g} max={float(probability_row['prob_max']):.6g} "
            f"mean={float(probability_row['prob_mean']):.6g} std={float(probability_row['prob_std']):.6g} "
            f"p99={float(probability_row['prob_p99']):.6g}"
        )
        print(f"  spatial_modes_used: {probability_row.get('mode_counts', '')}")
        if metrics_05 is not None:
            print(
                f"  metrics@0.5: IoU={float(metrics_05['iou']):.6g} Dice={float(metrics_05['dice']):.6g} "
                f"Precision={float(metrics_05['precision']):.6g} Recall={float(metrics_05['recall']):.6g} "
                f"pred_active_fraction={float(metrics_05['active_fraction_predicted']):.6g}"
            )
        print("")

    if warnings:
        print("Warnings")
        for warning in warnings:
            print(f"- {warning}")
        print("")

    print("Saved reports")
    for path in csv_paths:
        print(f"- {path}")
    print(f"- {visualization_root}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Diagnose multitask mask generalization across train, validation, and external test splits."
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path.")
    parser.add_argument(
        "--checkpoint-kind",
        choices=("best", "latest"),
        default="best",
        help="Which config-derived checkpoint to use when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Maximum chronological samples to analyze per split.",
    )
    parser.add_argument(
        "--num_visualizations",
        type=int,
        default=6,
        help="Maximum number of validation and external-test figures to save per split.",
    )
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_argument_parser().parse_args()
    if torch is None:
        raise ImportError("PyTorch is required for diagnose_mask_generalization.py.")
    if args.max_samples <= 0:
        raise ValueError(f"--max_samples must be positive, got {args.max_samples}.")
    if args.num_visualizations < 0:
        raise ValueError(f"--num_visualizations must be non-negative, got {args.num_visualizations}.")

    config = _ensure_config_path(load_config(args.config), args.config)
    if str(config.get("task_type", "regression")).lower() != "multitask":
        raise ValueError("diagnose_mask_generalization.py requires task_type: multitask.")
    if config.get("test_data_dir") in (None, "", "null"):
        raise ValueError(
            "No external test_data_dir configured. This diagnostic compares train/val against an external test dataset. "
            "Set test_data_dir in the config and rerun."
        )

    normalization_reference = _resolve_normalization_reference(config)
    datasets = _build_datasets(config, normalization_reference, max_samples=int(args.max_samples))
    if not datasets["external_test"]:
        raise ValueError("External test dataset could not be built.")

    input_channels = int(datasets["train"].total_input_channels)
    device = _get_device(config)
    model = build_model_from_config(config, input_channels=input_channels).to(device)

    checkpoint_path = _build_checkpoint_path(config, args.checkpoint, args.checkpoint_kind)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    visualization_root = Path("outputs/mask_diagnostics").expanduser().resolve()
    results = {
        split_name: diagnose_split(
            split_name=split_name,
            dataset=dataset,
            model=model,
            config=config,
            device=device,
            num_visualizations=int(args.num_visualizations),
            visualization_root=visualization_root,
        )
        for split_name, dataset in datasets.items()
    }

    source_rows = {split_name: payload["source_summary"] for split_name, payload in results.items()}
    probability_rows = {split_name: payload["probability_summary"] for split_name, payload in results.items()}
    threshold_rows_by_split = {split_name: payload["threshold_rows"] for split_name, payload in results.items()}
    fine_threshold_rows_by_split = {split_name: payload["fine_threshold_rows"] for split_name, payload in results.items()}
    warnings = _build_warnings(config, source_rows, probability_rows, threshold_rows_by_split)

    artifacts_root = Path("artifacts/logs").expanduser().resolve()
    source_csv_path = artifacts_root / "mask_source_distribution_by_split.csv"
    probability_csv_path = artifacts_root / "mask_probability_distribution_by_split.csv"
    threshold_csv_path = artifacts_root / "mask_threshold_sweep_metrics.csv"
    fine_threshold_csv_path = artifacts_root / "mask_fine_threshold_sweep_metrics.csv"
    _save_csv(list(source_rows.values()), source_csv_path)
    _save_csv(list(probability_rows.values()), probability_csv_path)
    threshold_rows_flat: list[dict[str, Any]] = []
    for split_rows in threshold_rows_by_split.values():
        threshold_rows_flat.extend(split_rows)
    _save_csv(threshold_rows_flat, threshold_csv_path)
    fine_threshold_rows_flat: list[dict[str, Any]] = []
    for split_rows in fine_threshold_rows_by_split.values():
        fine_threshold_rows_flat.extend(split_rows)
    _save_csv(fine_threshold_rows_flat, fine_threshold_csv_path)

    _print_summary(
        config=config,
        source_rows=source_rows,
        probability_rows=probability_rows,
        threshold_rows_by_split=threshold_rows_by_split,
        warnings=warnings,
        csv_paths=[source_csv_path, probability_csv_path, threshold_csv_path, fine_threshold_csv_path],
        visualization_root=visualization_root,
    )


if __name__ == "__main__":
    main()
