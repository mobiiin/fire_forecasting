"""Inspect the energy release target distribution for a dataset split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - environment-specific fallback
    matplotlib = None
    plt = None

try:
    import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
    torch = None

from src.config import load_config
from src.data.dataset import create_dataloaders, metadata_batch_to_list
from src.data.energy_release import inverse_transform_energy_target, resolve_energy_release_config


def _ensure_config_path(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    """Attach the config path so downstream helpers can resolve relative paths."""

    resolved_path = Path(config_path).expanduser().resolve()
    config = dict(config)
    config["config_path"] = str(resolved_path)
    config["_config_path"] = str(resolved_path)
    return config


def _select_loader(loaders: tuple[Any, Any, Any], split: str):
    """Return the loader for the requested split."""

    train_loader, val_loader, test_loader = loaders
    loader_by_split = {"train": train_loader, "val": val_loader, "test": test_loader}
    if split not in loader_by_split:
        raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}.")
    return loader_by_split[split]


def _to_numpy_energy_map(y_tensor, config: Mapping[str, Any]) -> np.ndarray:
    """Extract the total energy release target from one sample and convert it to MW."""

    if torch is not None and torch.is_tensor(y_tensor):
        y_array = y_tensor.detach().cpu().numpy()
    else:
        y_array = np.asarray(y_tensor)
    if y_array.ndim != 3 or y_array.shape[0] < 4:
        raise ValueError(f"Expected multitask target shaped (>=4, H, W), got {y_array.shape}.")
    return inverse_transform_energy_target(y_array[3], config)


def _distribution_stats(values: np.ndarray, active_threshold_MW: float) -> dict[str, float]:
    """Compute summary statistics for a flat array."""

    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise ValueError("Cannot compute distribution statistics on an empty array.")
    percentiles = np.percentile(flat, [50.0, 90.0, 95.0, 99.0, 99.9])
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "p50": float(percentiles[0]),
        "p90": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "p99_9": float(percentiles[4]),
        "fraction_zero": float(np.mean(flat == 0.0)),
        "fraction_active": float(np.mean(flat > active_threshold_MW)),
    }


def _print_stats(label: str, stats: Mapping[str, float]) -> None:
    """Print one stats block."""

    print(
        f"{label}: min={stats['min']:.6g} max={stats['max']:.6g} mean={stats['mean']:.6g} std={stats['std']:.6g} "
        f"p50={stats['p50']:.6g} p90={stats['p90']:.6g} p95={stats['p95']:.6g} p99={stats['p99']:.6g} "
        f"p99.9={stats['p99_9']:.6g} zero_frac={stats['fraction_zero']:.6g} active_frac={stats['fraction_active']:.6g}"
    )


def _save_histogram(values: np.ndarray, output_path: Path, title: str) -> None:
    """Save a histogram figure if matplotlib is available."""

    if plt is None:
        print("Matplotlib is unavailable; skipping histogram output.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    flat = values.reshape(-1)
    axes[0].hist(flat, bins=120, color="#cc5500")
    axes[0].set_title(f"{title} (MW)")
    axes[0].set_xlabel("MW")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.25)

    log_values = np.log1p(np.maximum(flat, 0.0))
    axes[1].hist(log_values, bins=120, color="#006d77")
    axes[1].set_title(f"{title} log1p(MW)")
    axes[1].set_xlabel("log1p(MW)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_example_maps(example_maps: list[tuple[str, np.ndarray]], output_dir: Path) -> None:
    """Save a few example energy release maps."""

    if plt is None:
        print("Matplotlib is unavailable; skipping example map output.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for example_name, energy_map in example_maps:
        vmax = float(np.percentile(energy_map, 99.0)) if np.any(np.isfinite(energy_map)) else 1.0
        vmax = max(vmax, 1.0e-6)
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(energy_map, cmap="inferno", vmin=0.0, vmax=vmax)
        ax.set_title(example_name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="MW")
        fig.tight_layout()
        fig.savefig(output_dir / f"{example_name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Inspect energy release target distributions.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split to inspect.")
    parser.add_argument("--max-examples", type=int, default=4, help="Maximum number of example maps to save.")
    args = parser.parse_args()

    config = _ensure_config_path(load_config(args.config), args.config)
    energy_release = resolve_energy_release_config(config)
    if not energy_release["enabled"]:
        raise ValueError("energy_release.enabled must be true to inspect the energy release target.")

    multitask = config.get("multitask", {}) if isinstance(config.get("multitask"), dict) else {}
    active_threshold_MW = float(multitask.get("energy_active_threshold_MW", 0.001))

    loader = _select_loader(create_dataloaders(config), args.split)
    if loader is None:
        raise ValueError(f"Requested split {args.split!r} is not available.")

    aggregate_maps: list[np.ndarray] = []
    per_dataset_maps: dict[str, list[np.ndarray]] = defaultdict(list)
    example_maps: list[tuple[str, np.ndarray]] = []

    for batch_index, batch in enumerate(loader):
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise TypeError("Expected DataLoader batches containing at least input and target tensors.")
        y_batch = batch[1]
        metadata_items = metadata_batch_to_list(batch[2]) if len(batch) >= 3 else [{} for _ in range(int(y_batch.shape[0]))]
        batch_size = int(y_batch.shape[0])
        for sample_index in range(batch_size):
            energy_map = _to_numpy_energy_map(y_batch[sample_index], config)
            aggregate_maps.append(energy_map)
            metadata = metadata_items[sample_index] if sample_index < len(metadata_items) else {}
            dataset_name = str(metadata.get("dataset_name", metadata.get("data_dir", "dataset")))
            per_dataset_maps[dataset_name].append(energy_map)
            if len(example_maps) < args.max_examples:
                example_maps.append((f"{dataset_name}_sample_{batch_index:04d}_{sample_index:02d}", energy_map))

    if not aggregate_maps:
        raise ValueError("No samples were found for the requested split.")

    aggregate_values = np.concatenate([energy_map.reshape(-1) for energy_map in aggregate_maps], axis=0)
    aggregate_stats = _distribution_stats(aggregate_values, active_threshold_MW)
    print(f"Split: {args.split}")
    _print_stats("energy_release_total_MW", aggregate_stats)
    log_stats = _distribution_stats(np.log1p(np.maximum(aggregate_values, 0.0)), math.inf)
    _print_stats("log1p_energy_release_total_MW", log_stats)

    if len(per_dataset_maps) > 1:
        print("")
        print("Per-dataset distributions:")
        for dataset_name in sorted(per_dataset_maps):
            dataset_values = np.concatenate([energy_map.reshape(-1) for energy_map in per_dataset_maps[dataset_name]], axis=0)
            _print_stats(dataset_name, _distribution_stats(dataset_values, active_threshold_MW))

    output_root = Path("outputs/diagnostics").resolve()
    histogram_path = output_root / f"energy_release_distribution_{args.split}.png"
    example_dir = output_root / f"energy_release_examples_{args.split}"
    _save_histogram(aggregate_values, histogram_path, f"Energy release {args.split}")
    _save_example_maps(example_maps, example_dir)
    print(f"Saved histogram: {histogram_path}")
    print(f"Saved example maps: {example_dir}")


if __name__ == "__main__":
    main()
