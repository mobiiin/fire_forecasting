"""Inspect selected dataset channels at specific timesteps and save 2D maps."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config


TAIL_CHANNEL_NAMES = [
    "flux_sensible_surface",
    "flux_latent_surface",
    "flux_sensible_canopy",
    "flux_latent_canopy",
    "fuel_surface",
    "fuel_canopy",
]


def resolve_path(base_path: Path, configured_path: str | Path) -> Path:
    """Resolve config-relative paths."""

    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_path.parent / path).resolve()


def extract_numeric_suffix(name: str) -> int | None:
    """Extract a trailing numeric suffix from a filename stem if present."""

    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else None


def sort_chronologically(file_paths: Iterable[Path]) -> list[Path]:
    """Sort files by trailing numeric suffix when available."""

    file_paths = list(file_paths)
    numeric_suffixes = [extract_numeric_suffix(path.stem) for path in file_paths]
    if all(value is not None for value in numeric_suffixes):
        return [path for _, path in sorted(zip(numeric_suffixes, file_paths), key=lambda item: item[0])]
    return sorted(file_paths, key=lambda path: path.name)


def discover_files(config_path: Path, config: dict) -> list[Path]:
    """Find dataset files from the config."""

    if "data_dir" not in config:
        raise KeyError("Config is missing required key 'data_dir'.")
    if "file_pattern" not in config:
        raise KeyError("Config is missing required key 'file_pattern'.")

    data_dir = resolve_path(config_path, config["data_dir"])
    file_pattern = str(config["file_pattern"])
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = sort_chronologically(data_dir.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found in '{data_dir}' using pattern '{file_pattern}'.")
    return files


def load_tensor(file_path: Path) -> np.ndarray:
    """Load one tensor from disk."""

    return np.load(file_path, mmap_mode="r", allow_pickle=False)


def channel_label(channel_index: int, total_channels: int) -> str:
    """Return a readable channel label for the tail channels when available."""

    tail_start = max(0, total_channels - len(TAIL_CHANNEL_NAMES))
    if channel_index >= tail_start:
        tail_offset = channel_index - tail_start
        if 0 <= tail_offset < len(TAIL_CHANNEL_NAMES):
            return TAIL_CHANNEL_NAMES[tail_offset]
    return f"channel_{channel_index:02d}"


def channel_group(channel_index: int, total_channels: int) -> str:
    """Group related channels so they can share colorbars."""

    tail_start = max(0, total_channels - len(TAIL_CHANNEL_NAMES))
    if tail_start <= channel_index <= tail_start + 3:
        return "flux"
    if tail_start + 4 <= channel_index <= tail_start + 5:
        return "fuel"
    return f"channel_{channel_index}"


def finite_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    """Compute stable display limits across a set of arrays."""

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


def channel_stats(channel_map: np.ndarray, active_threshold: float) -> dict[str, float | int]:
    """Compute summary statistics for one 2D channel map."""

    finite_mask = np.isfinite(channel_map)
    finite_values = channel_map[finite_mask]
    nan_count = int(np.isnan(channel_map).sum())
    inf_count = int(np.isinf(channel_map).sum())
    active_pixels = int(np.logical_and(finite_mask, channel_map > active_threshold).sum())

    if finite_values.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "active_pixels": active_pixels,
            "nan_count": nan_count,
            "inf_count": inf_count,
        }

    return {
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "mean": float(finite_values.mean()),
        "std": float(finite_values.std()),
        "active_pixels": active_pixels,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def save_channel_figure(
    tensor: np.ndarray,
    file_path: Path,
    timestep_index: int,
    channels: list[int],
    active_threshold: float,
    output_dir: Path,
    cmap: str,
) -> Path:
    """Save a 2x3 figure for the selected channels at one timestep."""

    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor shape (H, W, C), got {tensor.shape} in {file_path}.")

    height, width, total_channels = map(int, tensor.shape)
    _ = (height, width)

    if len(channels) > 6:
        raise ValueError(f"At most 6 channels fit in the 2x3 layout, got {len(channels)}.")

    channel_maps = {channel: np.asarray(tensor[:, :, channel], dtype=np.float32) for channel in channels}

    grouped_channels: dict[str, list[int]] = {}
    for channel in channels:
        grouped_channels.setdefault(channel_group(channel, total_channels), []).append(channel)

    color_limits = {
        group_name: finite_limits([channel_maps[channel] for channel in group_channels])
        for group_name, group_channels in grouped_channels.items()
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150, constrained_layout=True)
    axes_flat = axes.flatten()
    group_images: dict[str, object] = {}
    group_axes: dict[str, list[object]] = {}

    print(f"\nTimestep index {timestep_index} | file {file_path.name}")
    for panel_index, axis in enumerate(axes_flat):
        if panel_index >= len(channels):
            axis.axis("off")
            continue

        channel = channels[panel_index]
        group_name = channel_group(channel, total_channels)
        vmin, vmax = color_limits[group_name]
        channel_map = channel_maps[channel]
        stats = channel_stats(channel_map, active_threshold)

        image = axis.imshow(channel_map, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(f"{channel_label(channel, total_channels)} (ch {channel})")
        axis.set_xticks([])
        axis.set_yticks([])

        if group_name not in group_images:
            group_images[group_name] = image
            group_axes[group_name] = [axis]
        else:
            group_axes[group_name].append(axis)

        print(
            f"  ch {channel:02d} | min={stats['min']:.6g} max={stats['max']:.6g} "
            f"mean={stats['mean']:.6g} std={stats['std']:.6g} "
            f"active_pixels>{active_threshold:g}={stats['active_pixels']} "
            f"nan_count={stats['nan_count']} inf_count={stats['inf_count']}"
        )

    for group_name, image in group_images.items():
        label = f"{group_name} scale"
        fig.colorbar(image, ax=group_axes[group_name], fraction=0.03, pad=0.02, label=label)

    fig.suptitle(
        f"{file_path.name} | dataset index {timestep_index} | channels {', '.join(str(channel) for channel in channels)}",
        fontsize=14,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    channel_suffix = "_".join(str(channel) for channel in channels)
    output_path = output_dir / f"timestep_{timestep_index:05d}_{file_path.stem}_channels_{channel_suffix}.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Inspect selected target channels and save 2D channel-map figures."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        required=True,
        help="One to six channel indices to visualize in a 2x3 grid.",
    )
    parser.add_argument(
        "--timesteps",
        nargs="+",
        type=int,
        required=True,
        help="Dataset indices into the chronologically sorted tensor files.",
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=None,
        help="Threshold used to count active pixels. Defaults to config active_threshold or fire_threshold.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for saved figures.",
    )
    parser.add_argument(
        "--cmap",
        default=None,
        help="Optional matplotlib colormap. Defaults to config visualization.cmap or inferno.",
    )
    return parser


def main() -> None:
    """Run the channel inspection workflow."""

    args = build_argument_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    files = discover_files(config_path, config)

    channels = [int(channel) for channel in args.channels]
    if not channels:
        raise ValueError("At least one channel must be provided.")
    if len(channels) > 6:
        raise ValueError("This script supports at most 6 channels per figure.")
    if len(set(channels)) != len(channels):
        raise ValueError("Duplicate channel indices are not allowed.")

    timesteps = [int(timestep) for timestep in args.timesteps]
    if not timesteps:
        raise ValueError("At least one timestep must be provided.")

    visualization_config = config.get("visualization", {}) if isinstance(config.get("visualization"), dict) else {}
    default_threshold = float(config.get("active_threshold", config.get("fire_threshold", 0.0)))
    active_threshold = default_threshold if args.active_threshold is None else float(args.active_threshold)
    cmap = args.cmap or str(visualization_config.get("cmap", "inferno"))

    if args.output_dir is None:
        base_output_dir = resolve_path(
            config_path,
            visualization_config.get("output_path", "./outputs/visualizations"),
        )
        output_dir = base_output_dir / "target_channel_inspection"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()

    first_tensor = load_tensor(files[0])
    if first_tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensors shaped (H, W, C), got {first_tensor.shape} in {files[0]}.")
    total_channels = int(first_tensor.shape[2])

    invalid_channels = [channel for channel in channels if channel < 0 or channel >= total_channels]
    if invalid_channels:
        raise ValueError(
            f"Channel indices out of range for tensors with {total_channels} channels: {invalid_channels}"
        )

    invalid_timesteps = [timestep for timestep in timesteps if timestep < 0 or timestep >= len(files)]
    if invalid_timesteps:
        raise ValueError(
            f"Timestep indices out of range for dataset of {len(files)} files: {invalid_timesteps}"
        )

    print(f"Discovered {len(files)} files")
    print(f"Inspecting channels: {channels}")
    print(f"Inspecting dataset indices: {timesteps}")
    print(f"Active threshold: {active_threshold}")
    print(f"Output directory: {output_dir}")

    saved_paths = []
    for timestep_index in timesteps:
        file_path = files[timestep_index]
        tensor = np.asarray(load_tensor(file_path), dtype=np.float32)
        if tensor.shape[2] != total_channels:
            raise ValueError(
                f"Tensor channel count changed unexpectedly for {file_path}: "
                f"expected {total_channels}, got {tensor.shape[2]}"
            )

        saved_paths.append(
            save_channel_figure(
                tensor=tensor,
                file_path=file_path,
                timestep_index=timestep_index,
                channels=channels,
                active_threshold=active_threshold,
                output_dir=output_dir,
                cmap=cmap,
            )
        )

    print("\nSaved figures:")
    for saved_path in saved_paths:
        print(f"  {saved_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI safeguard
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
