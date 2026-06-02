"""Compare active-flux and burned-fuel mask definitions across train, val, and external test splits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
    torch = None

from scripts.diagnose_mask_generalization import (
    _all_mask_definitions_from_raw_sample,
    _build_datasets,
    _build_loader,
    _ensure_config_path,
    _extract_first_item,
    _resolve_normalization_reference,
    _save_csv,
    _select_plot_ordinals,
    _segmentation_counts,
)
from src.config import load_config

EPS = 1.0e-6


def _metadata_to_dict(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Convert collated batch metadata to simple Python values."""

    return {key: _extract_first_item(value) for key, value in metadata.items()}


def _save_mask_definition_plot(
    split_name: str,
    output_path: Path,
    sample_index: int,
    active_flux_source: np.ndarray,
    active_flux_mask: np.ndarray,
    burned_fuel_source: np.ndarray,
    burned_fuel_mask: np.ndarray,
) -> None:
    """Save a comparison figure for one sample."""

    active_flux_source = np.asarray(active_flux_source, dtype=np.float32)
    active_flux_mask = np.asarray(active_flux_mask, dtype=np.float32)
    burned_fuel_source = np.asarray(burned_fuel_source, dtype=np.float32)
    burned_fuel_mask = np.asarray(burned_fuel_mask, dtype=np.float32)
    agreement_map = active_flux_mask + 2.0 * burned_fuel_mask

    flux_vmin = float(np.nanmin(active_flux_source))
    flux_vmax = float(np.nanmax(active_flux_source))
    consumed_vmin = float(np.nanmin(burned_fuel_source))
    consumed_vmax = float(np.nanmax(burned_fuel_source))
    if np.isclose(flux_vmin, flux_vmax):
        flux_vmax = flux_vmin + 1.0
    if np.isclose(consumed_vmin, consumed_vmax):
        consumed_vmax = consumed_vmin + 1.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150, constrained_layout=True)
    panels = [
        ("Future flux source", active_flux_source, "inferno", flux_vmin, flux_vmax),
        ("Active-flux mask", active_flux_mask, "viridis", 0.0, 1.0),
        ("Combined consumed fuel source", burned_fuel_source, "magma", consumed_vmin, consumed_vmax),
        ("Burned-fuel mask", burned_fuel_mask, "viridis", 0.0, 1.0),
        ("Agreement map", agreement_map, "tab10", 0.0, 3.0),
        ("Contour overlay", active_flux_source, "inferno", flux_vmin, flux_vmax),
    ]

    for axis, (panel_title, panel_data, cmap, vmin, vmax) in zip(axes.flatten(), panels, strict=True):
        image = axis.imshow(panel_data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(panel_title)
        axis.set_xticks([])
        axis.set_yticks([])
        if panel_title == "Contour overlay":
            if np.nanmin(active_flux_mask) <= 0.5 <= np.nanmax(active_flux_mask):
                axis.contour(active_flux_mask, levels=[0.5], colors=["cyan"], linewidths=1.8)
            if np.nanmin(burned_fuel_mask) <= 0.5 <= np.nanmax(burned_fuel_mask):
                axis.contour(burned_fuel_mask, levels=[0.5], colors=["white"], linewidths=1.8, linestyles=["--"])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.suptitle(f"{split_name} sample_index={sample_index} | active_flux vs burned_fuel masks", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def compare_split(split_name: str, dataset, num_visualizations: int, output_root: Path) -> dict[str, Any]:
    """Compare the two raw mask definitions on one split."""

    loader = _build_loader(dataset)
    plot_ordinals = _select_plot_ordinals(len(dataset), num_visualizations)

    active_flux_pixels = 0
    burned_fuel_pixels = 0
    total_pixels = 0
    active_flux_nonempty_samples = 0
    burned_fuel_nonempty_samples = 0
    agreement_tp = 0.0
    agreement_fp = 0.0
    agreement_fn = 0.0

    for ordinal, batch in enumerate(loader):
        if not isinstance(batch, (tuple, list)) or len(batch) < 3:
            raise TypeError("Expected batches to contain input tensor, target tensor, and metadata.")
        metadata = _metadata_to_dict(batch[2])
        definitions = _all_mask_definitions_from_raw_sample(dataset, metadata, dataset.config)
        active_flux_source = np.asarray(definitions["active_flux"]["source_map"], dtype=np.float32)
        active_flux_mask = np.asarray(definitions["active_flux"]["mask"], dtype=np.float32)
        burned_fuel_source = np.asarray(definitions["burned_fuel"]["source_map"], dtype=np.float32)
        burned_fuel_mask = np.asarray(definitions["burned_fuel"]["mask"], dtype=np.float32)

        total_pixels += int(active_flux_mask.size)
        active_flux_pixels += int(np.sum(active_flux_mask))
        burned_fuel_pixels += int(np.sum(burned_fuel_mask))
        active_flux_nonempty_samples += int(np.any(active_flux_mask > 0.5))
        burned_fuel_nonempty_samples += int(np.any(burned_fuel_mask > 0.5))

        true_positive, false_positive, false_negative = _segmentation_counts(active_flux_mask, burned_fuel_mask)
        agreement_tp += true_positive
        agreement_fp += false_positive
        agreement_fn += false_negative

        if ordinal in plot_ordinals:
            sample_index = int(metadata.get("sample_index", ordinal))
            output_path = output_root / split_name / f"sample_{sample_index:05d}.png"
            _save_mask_definition_plot(
                split_name=split_name,
                output_path=output_path,
                sample_index=sample_index,
                active_flux_source=active_flux_source,
                active_flux_mask=active_flux_mask,
                burned_fuel_source=burned_fuel_source,
                burned_fuel_mask=burned_fuel_mask,
            )

    sample_count = int(len(dataset))
    iou = agreement_tp / (agreement_tp + agreement_fp + agreement_fn + EPS)
    dice = (2.0 * agreement_tp) / (2.0 * agreement_tp + agreement_fp + agreement_fn + EPS)

    return {
        "split": split_name,
        "sample_count": sample_count,
        "pixel_count": int(total_pixels),
        "active_flux_active_fraction": float(active_flux_pixels / max(total_pixels, 1)),
        "burned_fuel_active_fraction": float(burned_fuel_pixels / max(total_pixels, 1)),
        "active_flux_nonempty_samples": int(active_flux_nonempty_samples),
        "burned_fuel_nonempty_samples": int(burned_fuel_nonempty_samples),
        "active_flux_nonempty_sample_fraction": float(active_flux_nonempty_samples / max(sample_count, 1)),
        "burned_fuel_nonempty_sample_fraction": float(burned_fuel_nonempty_samples / max(sample_count, 1)),
        "mask_overlap_iou": float(iou),
        "mask_overlap_dice": float(dice),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description="Compare active-flux and burned-fuel mask definitions.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML config file.")
    parser.add_argument("--max_samples", type=int, default=200, help="Maximum chronological samples per split.")
    parser.add_argument("--num_visualizations", type=int, default=6, help="Maximum figures to save per split.")
    return parser


def main() -> None:
    """CLI entry point."""

    if torch is None:
        raise ImportError("PyTorch is required for compare_mask_definitions.py.")

    args = build_argument_parser().parse_args()
    config = _ensure_config_path(load_config(args.config), args.config)
    if config.get("test_data_dir") in (None, "", "null"):
        raise ValueError("compare_mask_definitions.py requires test_data_dir to compare the external test split.")

    datasets = _build_datasets(config, _resolve_normalization_reference(config), int(args.max_samples))
    output_root = Path("outputs/mask_definition_comparison").expanduser().resolve()
    rows = [
        compare_split(split_name, dataset, int(args.num_visualizations), output_root)
        for split_name, dataset in datasets.items()
    ]

    csv_path = Path("artifacts/logs/mask_definition_comparison.csv").expanduser().resolve()
    _save_csv(rows, csv_path)

    print("Mask Definition Comparison")
    for row in rows:
        print(f"[{row['split']}]")
        print(
            f"  active_flux_active_fraction={row['active_flux_active_fraction']:.6g} "
            f"burned_fuel_active_fraction={row['burned_fuel_active_fraction']:.6g}"
        )
        print(
            f"  mask_overlap_iou={row['mask_overlap_iou']:.6g} "
            f"mask_overlap_dice={row['mask_overlap_dice']:.6g}"
        )
        print(
            f"  active_flux_nonempty_samples={row['active_flux_nonempty_samples']}/{row['sample_count']} "
            f"burned_fuel_nonempty_samples={row['burned_fuel_nonempty_samples']}/{row['sample_count']}"
        )
        print("")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figures: {output_root}")


if __name__ == "__main__":
    main()
