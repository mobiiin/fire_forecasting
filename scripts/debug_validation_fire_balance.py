#!/usr/bin/env python3
"""Diagnose fire/no-fire membership on the exact configured validation loader."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from scripts.check_patch_fire_balance import classify_patch, get_patch, get_target_path, load_fire_mask
from src.config import load_config
from src.data.dataset import create_dataloaders
from src.evaluation.no_fire_metrics import FIRE_MASK_THRESHOLD, classify_target_masks
from src.training.train import resolve_validation_policy, validation_batch_indices_for_epoch, validation_loader_for_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved or source YAML configuration.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional prefix limit after loader selection.")
    parser.add_argument("--full-validation", action="store_true", help="Ignore the training screening subset and scan the full split.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def apply_loader_overrides(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    updated["return_metadata"] = True
    training = dict(updated.get("training", {}))
    data_loader = dict(updated.get("data_loader", {}))
    if args.batch_size is not None:
        if int(args.batch_size) <= 0:
            raise ValueError("--batch-size must be positive.")
        updated["batch_size"] = int(args.batch_size)
        training["batch_size"] = int(args.batch_size)
        data_loader["batch_size"] = int(args.batch_size)
    if int(args.num_workers) < 0:
        raise ValueError("--num-workers must be nonnegative.")
    training["num_workers"] = int(args.num_workers)
    training["persistent_workers"] = False
    data_loader["num_workers"] = int(args.num_workers)
    data_loader["persistent_workers"] = False
    updated["training"] = training
    updated["data_loader"] = data_loader
    return updated


def dataset_root_and_pattern(config: Mapping[str, Any]) -> tuple[Path, str]:
    dataloader = config.get("dataloader", {})
    processed = config.get("processed_dataset", {})
    if not isinstance(dataloader, Mapping) or str(dataloader.get("source", "")).lower() != "processed_full_frames":
        raise ValueError("This diagnostic currently requires dataloader.source=processed_full_frames.")
    root = Path(str(dataloader.get("dataset_root", processed.get("root")))).expanduser().resolve()
    return root, str(dataloader.get("sample_pattern", "consecutive5_h10"))


def select_loader(loaders: tuple[Any, Any, Any], split: str):
    return {"train": loaders[0], "val": loaders[1], "test": loaders[2]}[split]


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and int(args.max_samples) <= 0:
        raise ValueError("--max-samples must be positive.")
    config = apply_loader_overrides(load_config(args.config), args)
    root, pattern = dataset_root_and_pattern(config)
    loaders = create_dataloaders(config)
    loader = select_loader(loaders, args.split)

    selected_batches: list[int] | None = None
    selected_samples: list[int] | None = None
    scope = "full"
    if args.split == "val" and not args.full_validation:
        policy = resolve_validation_policy(config, val_loader=loader)
        selected_loader, selected_samples = validation_loader_for_epoch(loader, policy, epoch_number=1)
        selected_batches = validation_batch_indices_for_epoch(policy, epoch_number=1)
        scope = str(policy.get("validation_scope", "screening_subset"))
    else:
        selected_loader = loader

    dataset = getattr(selected_loader, "dataset", None)
    base_dataset = getattr(dataset, "dataset", dataset)
    records = getattr(base_dataset, "records", None)
    if records is None:
        raise TypeError("Diagnostic requires a processed validation dataset exposing ordered sample records.")
    if selected_samples is not None:
        record_indices = [int(index) for index in selected_samples]
    elif selected_batches is not None:
        batch_size = int(getattr(loader, "batch_size", 1) or 1)
        record_indices = [
            sample_index
            for batch_index in selected_batches
            for sample_index in range(batch_index * batch_size, min((batch_index + 1) * batch_size, len(records)))
        ]
    else:
        record_indices = list(range(len(records)))
    if args.max_samples is not None:
        record_indices = record_indices[: int(args.max_samples)]

    @lru_cache(maxsize=16)
    def cached_mask(path_text: str):
        return load_fire_mask(path_text)

    loader_total = loader_fire = loader_no_fire = 0
    canonical_total = canonical_fire = canonical_no_fire = 0
    first_no_fire_ids: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for position, record_index in enumerate(record_indices, start=1):
        item = dict(records[record_index])
        target_path = get_target_path(item, root, sample_pattern=pattern).resolve()
        fire_mask = cached_mask(str(target_path))
        patch = get_patch(item)
        y0, x0, height, width = (int(patch[key]) for key in ("y0", "x0", "height", "width"))
        # Reproduce ProcessedTemporalPatchDataset target channel 2 exactly:
        # crop the indexed target first, then binarize with a strict > 0.5.
        cropped = (fire_mask[y0 : y0 + height, x0 : x0 + width] > FIRE_MASK_THRESHOLD).astype(np.float32)
        target = torch.zeros(1, 4, height, width, dtype=torch.float32)
        target[0, 2] = torch.from_numpy(cropped)
        loader_classification = classify_target_masks(target, fire_threshold=FIRE_MASK_THRESHOLD)
        loader_has_fire = bool(loader_classification["has_fire"].item())
        loader_active_pixels = int(loader_classification["active_pixels"].item())
        loader_total += 1
        loader_fire += int(loader_has_fire)
        loader_no_fire += int(not loader_has_fire)

        canonical = classify_patch(
            fire_mask,
            patch,
            threshold=FIRE_MASK_THRESHOLD,
            active_fraction_threshold=0.0,
        )
        canonical_total += 1
        canonical_fire += int(bool(canonical["has_fire"]))
        canonical_no_fire += int(not bool(canonical["has_fire"]))
        sample_id = str(item.get("sample_id", f"record_{record_index}"))
        if not loader_has_fire and len(first_no_fire_ids) < 20:
            first_no_fire_ids.append(sample_id)
        if loader_has_fire != bool(canonical["has_fire"]) or loader_active_pixels != int(canonical["active_pixels"]):
            mismatches.append({
                "sample_id": sample_id,
                "loader_has_fire": loader_has_fire,
                "canonical_has_fire": bool(canonical["has_fire"]),
                "loader_active_pixels": loader_active_pixels,
                "canonical_active_pixels": int(canonical["active_pixels"]),
            })
            if len(mismatches) >= 20:
                break
        if position == 1 or position % 1000 == 0 or position == len(record_indices):
            print(f"Checked {position}/{len(record_indices)} exact loader records...", flush=True)

    report = {
        "config": str(Path(args.config).expanduser().resolve()),
        "split": args.split,
        "scope": scope,
        "full_validation": bool(args.full_validation),
        "max_samples": args.max_samples,
        "fire_threshold": FIRE_MASK_THRESHOLD,
        "loader_count_strategy": "exact configured loader dataset records with ProcessedTemporalPatchDataset target crop/binarization; input frames intentionally not decompressed",
        "canonical_definition": "fire iff any target mask channel-2 pixel is strictly > 0.5; no-fire iff active-pixel count is zero",
        "loader_counts": {
            "total": loader_total,
            "fire": loader_fire,
            "no_fire": loader_no_fire,
            "fire_percent": 100.0 * loader_fire / loader_total if loader_total else 0.0,
            "no_fire_percent": 100.0 * loader_no_fire / loader_total if loader_total else 0.0,
        },
        "canonical_counts": {
            "total": canonical_total,
            "fire": canonical_fire,
            "no_fire": canonical_no_fire,
            "fire_percent": 100.0 * canonical_fire / canonical_total if canonical_total else 0.0,
            "no_fire_percent": 100.0 * canonical_no_fire / canonical_total if canonical_total else 0.0,
        },
        "counts_match": (loader_total, loader_fire, loader_no_fire) == (canonical_total, canonical_fire, canonical_no_fire),
        "first_20_no_fire_sample_ids": first_no_fire_ids,
        "mismatches": mismatches,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if mismatches or not report["counts_match"]:
        raise RuntimeError("Validation-loader classification disagrees with the canonical patch-balance classifier.")
    if canonical_no_fire > 0 and loader_no_fire == 0:
        raise RuntimeError("Canonical checker found no-fire patches but loader-based evaluation counted zero.")


if __name__ == "__main__":
    main()
