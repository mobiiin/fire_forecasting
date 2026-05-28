"""Evaluate a persistence baseline on the configured wildfire test split."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.data.splits import chronological_split_indices


def resolve_path(base_path: Path, configured_path: str | Path) -> Path:
    """Resolve a configured path relative to the config location."""

    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_path.parent / path).resolve()


def extract_numeric_suffix(name: str) -> int | None:
    """Extract a trailing numeric suffix from a filename stem."""

    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else None


def sort_chronologically(file_paths: Sequence[Path]) -> list[Path]:
    """Sort files by numeric suffix when available, otherwise lexicographically."""

    numeric_suffixes = [extract_numeric_suffix(path.stem) for path in file_paths]
    if all(value is not None for value in numeric_suffixes):
        return [path for _, path in sorted(zip(numeric_suffixes, file_paths), key=lambda item: item[0])]
    return sorted(file_paths, key=lambda path: path.name)


def discover_files(config_path: Path, config: Mapping[str, Any]) -> list[Path]:
    """Resolve and discover dataset files from the config."""

    if "data_dir" not in config:
        raise KeyError("Config is missing required key 'data_dir'.")
    if "file_pattern" not in config:
        raise KeyError("Config is missing required key 'file_pattern'.")

    data_dir = resolve_path(config_path, config["data_dir"])
    file_pattern = str(config["file_pattern"])
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = sort_chronologically(list(data_dir.glob(file_pattern)))
    if not files:
        raise FileNotFoundError(f"No files found in '{data_dir}' using pattern '{file_pattern}'.")
    return files


def load_tensor(file_path: Path) -> np.ndarray:
    """Load one tensor file from disk."""

    return np.load(file_path, mmap_mode="r", allow_pickle=False)


def _get_section(config: Mapping[str, Any], *names: str) -> dict[str, Any]:
    """Return the first nested mapping found under any of the provided names."""

    for name in names:
        section = config.get(name)
        if isinstance(section, dict):
            return section
    return {}


def warning_messages_for_target_channel(
    target_channel: int,
    input_channel_count: int,
) -> list[str]:
    """Return likely-target warnings based on channel selection."""

    warnings: list[str] = []
    if target_channel in {54, 55}:
        warnings.append(
            "According to the dataset description, this may be a fuel load channel, not fire intensity."
        )
    if target_channel >= input_channel_count:
        warnings.append(
            "The model does not see the previous target channel in its input history."
        )
    return warnings


def select_sample_indices(sample_count: int, maximum_samples: int) -> list[int]:
    """Choose evenly spaced sample positions for saved visualizations."""

    if sample_count <= 0 or maximum_samples <= 0:
        return []
    if sample_count <= maximum_samples:
        return list(range(sample_count))

    raw_indices = np.linspace(0, sample_count - 1, num=maximum_samples)
    return sorted({int(round(value)) for value in raw_indices})


def ensure_finite_map(channel_map: np.ndarray, file_path: Path, channel_index: int) -> np.ndarray:
    """Validate a 2D channel map before metric computation."""

    channel_map = np.asarray(channel_map, dtype=np.float32)
    if channel_map.ndim != 2:
        raise ValueError(
            f"Expected a 2D target map for channel {channel_index} in {file_path}, got {channel_map.shape}."
        )
    nan_count = int(np.isnan(channel_map).sum())
    inf_count = int(np.isinf(channel_map).sum())
    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"Target map for channel {channel_index} in {file_path} contains non-finite values: "
            f"nan_count={nan_count}, inf_count={inf_count}."
        )
    return channel_map


def crop_channel_map(
    channel_map: np.ndarray,
    patch_top: int | None = None,
    patch_left: int | None = None,
    patch_size: int | None = None,
) -> np.ndarray:
    """Crop a 2D channel map to match an evaluation patch when requested."""

    channel_map = np.asarray(channel_map, dtype=np.float32)
    if patch_top is None and patch_left is None and patch_size is None:
        return channel_map
    if patch_top is None or patch_left is None or patch_size is None:
        raise ValueError("patch_top, patch_left, and patch_size must be provided together.")

    patch_top = int(patch_top)
    patch_left = int(patch_left)
    patch_size = int(patch_size)
    patch_bottom = patch_top + patch_size
    patch_right = patch_left + patch_size
    cropped = channel_map[patch_top:patch_bottom, patch_left:patch_right]
    if cropped.shape != (patch_size, patch_size):
        raise ValueError(
            "Patch crop produced an unexpected shape. "
            f"Expected {(patch_size, patch_size)}, got {cropped.shape}."
        )
    return np.asarray(cropped, dtype=np.float32)


def build_persistence_sample(
    config: Mapping[str, Any],
    files: Sequence[Path],
    target_channel: int,
    sample_start: int,
    patch_top: int | None = None,
    patch_left: int | None = None,
    patch_size: int | None = None,
) -> dict[str, Any]:
    """Load the raw current/target maps for one test sample and form the persistence baseline."""

    input_last_index = int(sample_start) + int(config["input_sequence_length"]) - 1
    target_index = input_last_index + int(config["prediction_horizon"])
    current_file = Path(files[input_last_index]).expanduser().resolve()
    target_file = Path(files[target_index]).expanduser().resolve()

    current_tensor = load_tensor(current_file)
    target_tensor = load_tensor(target_file)
    current_map = ensure_finite_map(current_tensor[:, :, target_channel], current_file, target_channel)
    true_future_map = ensure_finite_map(target_tensor[:, :, target_channel], target_file, target_channel)

    current_map = crop_channel_map(current_map, patch_top=patch_top, patch_left=patch_left, patch_size=patch_size)
    true_future_map = crop_channel_map(
        true_future_map,
        patch_top=patch_top,
        patch_left=patch_left,
        patch_size=patch_size,
    )
    persistence_prediction = np.asarray(current_map, dtype=np.float32).copy()

    return {
        "sample_start": int(sample_start),
        "input_last_index": input_last_index,
        "target_index": target_index,
        "current_file": current_file,
        "target_file": target_file,
        "current_map": current_map,
        "true_future_map": true_future_map,
        "persistence_prediction": persistence_prediction,
    }


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    active_threshold: float,
    eps: float,
) -> dict[str, float]:
    """Compute the same regression metrics used in training, but in NumPy."""

    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    abs_error = np.abs(prediction - target)
    mae = float(abs_error.mean())
    rmse = float(np.sqrt(np.mean((prediction - target) ** 2) + eps))

    active_mask = target > active_threshold
    if np.any(active_mask):
        active_region_mae = float(abs_error[active_mask].mean())
    else:
        active_region_mae = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "active_mae": active_region_mae,
        "active_region_mae": active_region_mae,
    }


def threshold_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    prediction_threshold: float,
    target_threshold: float,
    eps: float,
) -> dict[str, float]:
    """Compute thresholded segmentation-style metrics in NumPy."""

    predicted_mask = np.asarray(prediction >= prediction_threshold, dtype=np.float32)
    target_mask = np.asarray(target >= target_threshold, dtype=np.float32)

    true_positive = float(np.sum(predicted_mask * target_mask))
    false_positive = float(np.sum(predicted_mask * (1.0 - target_mask)))
    false_negative = float(np.sum((1.0 - predicted_mask) * target_mask))

    iou = true_positive / (true_positive + false_positive + false_negative + eps)
    dice = (2.0 * true_positive) / (2.0 * true_positive + false_positive + false_negative + eps)
    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive / (true_positive + false_negative + eps)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


def summarize_target_distribution(
    maps: Sequence[np.ndarray],
    active_threshold: float,
) -> dict[str, float]:
    """Summarize target-map values across the evaluated split."""

    flattened = [np.asarray(channel_map, dtype=np.float32).reshape(-1) for channel_map in maps]
    if not flattened:
        raise ValueError("No target maps were provided for summary statistics.")

    values = np.concatenate(flattened)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        raise ValueError("Target maps contain no finite values.")

    return {
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "mean": float(finite_values.mean()),
        "std": float(finite_values.std()),
        "active_pixel_fraction": float(np.mean(finite_values > active_threshold)),
    }


def _finite_min_max(*arrays: np.ndarray) -> tuple[float, float]:
    """Compute a stable display range across multiple arrays."""

    finite_values = []
    for array in arrays:
        values = np.asarray(array, dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size > 0:
            finite_values.append(values)

    if not finite_values:
        return 0.0, 1.0

    merged = np.concatenate(finite_values)
    vmin = float(merged.min())
    vmax = float(merged.max())
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def save_persistence_figure(
    current_map: np.ndarray,
    true_future_map: np.ndarray,
    predicted_map: np.ndarray,
    output_path: Path,
    title: str,
    cmap: str,
    dpi: int,
) -> Path:
    """Save a 2x2 persistence-baseline diagnostic figure."""

    error_map = np.abs(predicted_map - true_future_map)
    shared_vmin, shared_vmax = _finite_min_max(current_map, true_future_map, predicted_map)
    error_vmin, error_vmax = _finite_min_max(error_map)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=dpi, constrained_layout=True)
    axes_flat = axes.flatten()
    panel_specs = [
        ("Current target map", current_map, cmap, shared_vmin, shared_vmax),
        ("True future target map", true_future_map, cmap, shared_vmin, shared_vmax),
        ("Persistence prediction", predicted_map, cmap, shared_vmin, shared_vmax),
        ("Absolute error", error_map, "magma", error_vmin, error_vmax),
    ]

    shared_images = []
    error_image = None
    for axis, (panel_title, panel_data, panel_cmap, vmin, vmax) in zip(axes_flat, panel_specs):
        image = axis.imshow(panel_data, origin="lower", cmap=panel_cmap, vmin=vmin, vmax=vmax)
        axis.set_title(panel_title)
        axis.set_xticks([])
        axis.set_yticks([])
        if panel_title == "Absolute error":
            error_image = image
        else:
            shared_images.append((axis, image))

    if shared_images:
        fig.colorbar(shared_images[0][1], ax=[axis for axis, _ in shared_images], fraction=0.03, pad=0.02)
    if error_image is not None:
        fig.colorbar(error_image, ax=axes_flat[3], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def evaluate_persistence_for_channel(
    config_path: str | Path,
    target_channel: int,
    num_visualizations: int = 5,
    compute_threshold_metrics: bool = False,
    threshold: float | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate persistence for one channel on the same test split used by the model."""

    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    files = discover_files(config_path, config)

    required_keys = (
        "input_sequence_length",
        "prediction_horizon",
        "train_fraction",
        "val_fraction",
        "test_fraction",
    )
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise KeyError(f"Config is missing required key(s): {', '.join(missing)}")

    first_tensor = load_tensor(files[0])
    if first_tensor.ndim != 3:
        raise ValueError(f"Expected dataset tensors shaped (H, W, C), got {first_tensor.shape} in {files[0]}.")
    total_channels = int(first_tensor.shape[2])
    if target_channel < 0 or target_channel >= total_channels:
        raise ValueError(
            f"target_channel out of range for tensors with {total_channels} channels: {target_channel}."
        )
    input_channel_count = int(config.get("input_channel_count", _get_section(config, "model").get("input_channels", 0)))

    splits = chronological_split_indices(
        num_timesteps=len(files),
        input_sequence_length=int(config["input_sequence_length"]),
        prediction_horizon=int(config["prediction_horizon"]),
        train_fraction=float(config["train_fraction"]),
        val_fraction=float(config["val_fraction"]),
        test_fraction=float(config["test_fraction"]),
    )
    test_indices = splits["test"]
    if not test_indices:
        raise ValueError("Test split is empty; cannot evaluate the persistence baseline.")

    active_threshold = float(config.get("active_threshold", config.get("fire_threshold", 0.0)))
    metrics_config = _get_section(config, "metrics")
    eps = float(metrics_config.get("eps", _get_section(config, "training").get("eps", 1e-6)))
    threshold_value = float(config.get("fire_threshold", 0.5)) if threshold is None else float(threshold)

    visualization_config = _get_section(config, "visualization")
    if output_dir is None:
        base_output_dir = resolve_path(
            config_path,
            visualization_config.get("output_path", "./outputs/visualizations"),
        )
        output_dir_path = base_output_dir / "persistence_baseline"
    else:
        output_dir_path = Path(output_dir).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    cmap = str(visualization_config.get("cmap", "inferno"))
    dpi = int(visualization_config.get("dpi", 150))

    regression_metric_totals: dict[str, float] = {}
    threshold_metric_totals: dict[str, float] = {}
    saved_paths: list[Path] = []
    target_maps: list[np.ndarray] = []
    visualization_positions = set(select_sample_indices(len(test_indices), int(num_visualizations)))

    if verbose:
        print(f"Discovered {len(files)} files")
        print(f"Target channel: {target_channel}")
        print(f"Test sample count: {len(test_indices)}")
        print(f"Active-region threshold: {active_threshold}")
        for warning_message in warning_messages_for_target_channel(target_channel, input_channel_count):
            print(f"WARNING: {warning_message}")
        if compute_threshold_metrics:
            print(f"Threshold metrics enabled at threshold={threshold_value}")

    for test_position, sample_start in enumerate(test_indices):
        sample_record = build_persistence_sample(
            config=config,
            files=files,
            target_channel=target_channel,
            sample_start=sample_start,
        )
        current_file = Path(sample_record["current_file"])
        target_file = Path(sample_record["target_file"])
        current_map = np.asarray(sample_record["current_map"], dtype=np.float32)
        true_future_map = np.asarray(sample_record["true_future_map"], dtype=np.float32)
        target_maps.append(true_future_map)
        persistence_prediction = np.asarray(sample_record["persistence_prediction"], dtype=np.float32)

        sample_regression_metrics = regression_metrics(
            prediction=persistence_prediction,
            target=true_future_map,
            active_threshold=active_threshold,
            eps=eps,
        )
        for metric_name, metric_value in sample_regression_metrics.items():
            regression_metric_totals[metric_name] = regression_metric_totals.get(metric_name, 0.0) + float(metric_value)

        if compute_threshold_metrics:
            sample_threshold_metrics = threshold_metrics(
                prediction=persistence_prediction,
                target=true_future_map,
                prediction_threshold=threshold_value,
                target_threshold=threshold_value,
                eps=eps,
            )
            for metric_name, metric_value in sample_threshold_metrics.items():
                threshold_metric_totals[metric_name] = threshold_metric_totals.get(metric_name, 0.0) + float(metric_value)

        if test_position in visualization_positions:
            figure_path = output_dir_path / f"sample_{test_position:05d}_start_{sample_start:05d}.png"
            saved_paths.append(
                save_persistence_figure(
                    current_map=current_map,
                    true_future_map=true_future_map,
                    predicted_map=persistence_prediction,
                    output_path=figure_path,
                    title=(
                        f"Persistence baseline | test sample {test_position:05d} | "
                        f"current {current_file.name} -> target {target_file.name} | ch {target_channel}"
                    ),
                    cmap=cmap,
                    dpi=dpi,
                )
            )

    averaged_regression_metrics = {
        f"test_{metric_name}": total / len(test_indices)
        for metric_name, total in regression_metric_totals.items()
    }
    averaged_threshold_metrics = {
        f"test_{metric_name}": total / len(test_indices)
        for metric_name, total in threshold_metric_totals.items()
    }
    target_distribution = summarize_target_distribution(target_maps, active_threshold)

    if verbose:
        print("\nPersistence baseline metrics:")
        for metric_name, metric_value in averaged_regression_metrics.items():
            print(f"  {metric_name}: {metric_value:.6f}")
        for metric_name, metric_value in averaged_threshold_metrics.items():
            print(f"  {metric_name}: {metric_value:.6f}")
        print(
            "  target distribution: "
            f"min={target_distribution['min']:.6g} max={target_distribution['max']:.6g} "
            f"mean={target_distribution['mean']:.6g} std={target_distribution['std']:.6g} "
            f"active_pixel_fraction={target_distribution['active_pixel_fraction']:.6f}"
        )

        if saved_paths:
            print("\nSaved visualizations:")
            for saved_path in saved_paths:
                print(f"  {saved_path}")

    return {
        "target_channel": target_channel,
        "input_channel_count": input_channel_count,
        "num_test_samples": len(test_indices),
        "regression_metrics": averaged_regression_metrics,
        "threshold_metrics": averaged_threshold_metrics,
        "target_distribution": target_distribution,
        "warnings": warning_messages_for_target_channel(target_channel, input_channel_count),
        "saved_paths": [str(path) for path in saved_paths],
    }


def evaluate_persistence_baseline(
    config_path: str | Path,
    num_visualizations: int = 5,
    compute_threshold_metrics: bool = False,
    threshold: float | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate persistence on the config-selected target channel."""

    config_path = Path(config_path).expanduser().resolve()
    config = load_config(config_path)
    if "target_channel" not in config:
        raise KeyError("Config is missing required key 'target_channel'.")
    return evaluate_persistence_for_channel(
        config_path=config_path,
        target_channel=int(config["target_channel"]),
        num_visualizations=num_visualizations,
        compute_threshold_metrics=compute_threshold_metrics,
        threshold=threshold,
        output_dir=output_dir,
        verbose=True,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Evaluate a persistence baseline on the configured wildfire test split."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=5,
        help="Number of evenly spaced persistence-baseline visualizations to save from the test split.",
    )
    parser.add_argument(
        "--compute-threshold-metrics",
        action="store_true",
        help="Also compute thresholded IoU/Dice/precision/recall using the provided threshold.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for optional segmentation-style metrics. Defaults to config fire_threshold.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for saved visualizations.",
    )
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_argument_parser().parse_args()
    evaluate_persistence_baseline(
        config_path=args.config,
        num_visualizations=args.num_visualizations,
        compute_threshold_metrics=bool(args.compute_threshold_metrics),
        threshold=args.threshold,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI safeguard
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
