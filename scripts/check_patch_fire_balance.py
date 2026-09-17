#!/usr/bin/env python3
"""Check fire/no-fire balance in processed patchified samples.

This script inspects the metadata-only temporal sample index and the cropped
target fire masks. It does not modify the dataset and does not load or run a
model.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SAMPLE_PATTERNS = ("consecutive5_h10", "single1_h10", "sparse5_h10")
SPLITS = ("train", "val", "test", "all")
BIN_ORDER = ("no_fire", "tiny_fire", "small_fire", "medium_fire", "large_fire")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Sample index not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}, got {type(payload).__name__}")
            records.append(payload)
    return records


def get_sample_split(sample: Mapping[str, Any]) -> str | None:
    value = sample.get("split")
    return str(value) if value is not None else None


def get_fire_name(sample: Mapping[str, Any]) -> str:
    for key in ("fire_name", "fire"):
        value = sample.get(key)
        if value:
            return str(value)
    raise KeyError(f"Could not determine fire name. Sample keys: {sorted(sample.keys())}")


def _int_from_any(value: Any, field: str) -> int:
    if value is None:
        raise KeyError(field)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Patch field {field!r} must be an integer, got {value!r}") from exc


def get_patch(sample: Mapping[str, Any]) -> dict[str, int]:
    patch = sample.get("patch")
    if isinstance(patch, Mapping):
        y0 = patch.get("y0", patch.get("y", patch.get("patch_y")))
        x0 = patch.get("x0", patch.get("x", patch.get("patch_x")))
        height = patch.get("height", patch.get("h", patch.get("patch_h")))
        width = patch.get("width", patch.get("w", patch.get("patch_w")))
        try:
            parsed = {
                "y0": _int_from_any(y0, "patch.y0"),
                "x0": _int_from_any(x0, "patch.x0"),
                "height": _int_from_any(height, "patch.height"),
                "width": _int_from_any(width, "patch.width"),
            }
        except KeyError:
            parsed = {}
        else:
            _validate_patch(parsed, sample)
            return parsed

    candidates = (
        ("patch_y", "patch_x", "patch_h", "patch_w"),
        ("y0", "x0", "height", "width"),
        ("y", "x", "h", "w"),
    )
    for y_key, x_key, h_key, w_key in candidates:
        if all(key in sample for key in (y_key, x_key, h_key, w_key)):
            parsed = {
                "y0": _int_from_any(sample[y_key], y_key),
                "x0": _int_from_any(sample[x_key], x_key),
                "height": _int_from_any(sample[h_key], h_key),
                "width": _int_from_any(sample[w_key], w_key),
            }
            _validate_patch(parsed, sample)
            return parsed

    raise KeyError(f"Could not determine patch coordinates. Sample keys: {sorted(sample.keys())}")


def _validate_patch(patch: Mapping[str, int], sample: Mapping[str, Any]) -> None:
    if patch["height"] <= 0 or patch["width"] <= 0:
        raise ValueError(f"Patch height/width must be positive, got {dict(patch)} for sample {sample.get('sample_id', '<unknown>')}")
    if patch["y0"] < 0 or patch["x0"] < 0:
        raise ValueError(f"Patch origin must be non-negative, got {dict(patch)} for sample {sample.get('sample_id', '<unknown>')}")


def get_target_path(sample: Mapping[str, Any], dataset_root: str | Path, sample_pattern: str | None = None) -> Path:
    root = Path(dataset_root).expanduser()
    for key in ("target_path", "target_file"):
        value = sample.get(key)
        if value:
            path = Path(str(value)).expanduser()
            return path if path.is_absolute() else root / path

    fire = get_fire_name(sample)
    current_index = sample.get("current_index")
    future_index = sample.get("future_index", sample.get("target_index"))
    horizon = sample.get("horizon")
    if horizon is None and sample_pattern and "_h" in sample_pattern:
        try:
            horizon = int(str(sample_pattern).rsplit("_h", 1)[1])
        except ValueError:
            horizon = None
    if current_index is None or future_index is None or horizon is None:
        raise KeyError(
            "Could not construct target path. Need target_path/target_file or "
            f"fire/current_index/future_index/horizon. Sample keys: {sorted(sample.keys())}"
        )
    return root / "targets" / f"h{int(horizon)}" / fire / f"target_current_{int(current_index):06d}_future_{int(future_index):06d}.npz"


def load_fire_mask(target_path: str | Path) -> np.ndarray:
    path = Path(target_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Target file not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        for key in ("fire_mask", "mask"):
            if key in archive.files:
                return np.asarray(archive[key])
        for key in ("y", "target"):
            if key in archive.files:
                array = np.asarray(archive[key])
                if array.ndim < 3 or array.shape[0] <= 2:
                    raise ValueError(f"Target array {key!r} in {path} cannot provide channel 2; shape={array.shape}")
                return np.asarray(array[2])
    raise KeyError(f"No fire mask found in {path}. Available keys: {archive.files if 'archive' in locals() else '<closed>'}")


def classify_patch(
    fire_mask: np.ndarray,
    patch: Mapping[str, int],
    threshold: float = 0.5,
    active_fraction_threshold: float = 0.0,
) -> dict[str, Any]:
    mask = np.asarray(fire_mask)
    if mask.ndim != 2:
        raise ValueError(f"Fire mask must be 2-D, got shape={mask.shape}")
    y0, x0, height, width = (int(patch[key]) for key in ("y0", "x0", "height", "width"))
    if y0 + height > mask.shape[0] or x0 + width > mask.shape[1]:
        raise ValueError(f"Patch {dict(patch)} is outside fire mask shape={mask.shape}")
    fire_patch = mask[y0 : y0 + height, x0 : x0 + width]
    active_pixels = int((fire_patch > float(threshold)).sum())
    total_pixels = int(height * width)
    active_fraction = float(active_pixels / total_pixels) if total_pixels else 0.0
    min_fraction = float(active_fraction_threshold)
    has_fire = active_fraction > 0.0 if min_fraction <= 0.0 else active_fraction >= min_fraction
    return {
        "active_pixels": active_pixels,
        "total_pixels": total_pixels,
        "active_fraction": active_fraction,
        "has_fire": bool(has_fire),
        "bin": active_fraction_bin(active_fraction),
    }


def active_fraction_bin(active_fraction: float) -> str:
    if active_fraction == 0.0:
        return "no_fire"
    if active_fraction < 0.001:
        return "tiny_fire"
    if active_fraction < 0.01:
        return "small_fire"
    if active_fraction < 0.05:
        return "medium_fire"
    return "large_fire"


def _empty_counts() -> dict[str, Any]:
    return {
        "total": 0,
        "fire": 0,
        "no_fire": 0,
        "bins": {name: 0 for name in BIN_ORDER},
        "active_pixels": 0,
        "total_pixels": 0,
    }


def _add_classification(counts: dict[str, Any], classification: Mapping[str, Any]) -> None:
    counts["total"] += 1
    if bool(classification["has_fire"]):
        counts["fire"] += 1
    else:
        counts["no_fire"] += 1
    counts["bins"][str(classification["bin"])] += 1
    counts["active_pixels"] += int(classification["active_pixels"])
    counts["total_pixels"] += int(classification["total_pixels"])


def check_balance(
    dataset_root: str | Path,
    sample_pattern: str = "consecutive5_h10",
    split: str = "train",
    fire_threshold: float = 0.5,
    active_fraction_threshold: float = 0.0,
    max_samples: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    sample_index = root / "indices" / "temporal" / f"samples_{sample_pattern}.jsonl"
    records = load_jsonl(sample_index)
    if split != "all":
        records = [record for record in records if get_sample_split(record) == split]
    if max_samples is not None:
        records = records[: int(max_samples)]
    if not records:
        raise ValueError(f"No samples found in {sample_index} for split={split!r}")

    totals = _empty_counts()
    per_fire: dict[str, dict[str, Any]] = defaultdict(_empty_counts)

    for index, sample in enumerate(records, start=1):
        target_path = get_target_path(sample, root, sample_pattern=sample_pattern)
        patch = get_patch(sample)
        fire_name = get_fire_name(sample)
        fire_mask = load_fire_mask(target_path)
        classification = classify_patch(
            fire_mask,
            patch,
            threshold=fire_threshold,
            active_fraction_threshold=active_fraction_threshold,
        )
        _add_classification(totals, classification)
        _add_classification(per_fire[fire_name], classification)
        if verbose and (index == 1 or index % 1000 == 0 or index == len(records)):
            print(f"Checked {index}/{len(records)} samples...", flush=True)

    summary = {
        "dataset_root": str(root),
        "sample_index": str(sample_index),
        "sample_pattern": sample_pattern,
        "split": split,
        "fire_threshold": float(fire_threshold),
        "active_fraction_threshold": float(active_fraction_threshold),
        "counts": finalize_counts(totals),
        "per_fire": {fire: finalize_counts(counts) for fire, counts in sorted(per_fire.items())},
    }
    summary["diagnosis"] = diagnose_balance(summary["counts"]["fire_percent"])
    return summary


def finalize_counts(counts: Mapping[str, Any]) -> dict[str, Any]:
    total = int(counts["total"])
    fire = int(counts["fire"])
    no_fire = int(counts["no_fire"])
    total_pixels = int(counts.get("total_pixels", 0))
    active_pixels = int(counts.get("active_pixels", 0))

    def pct(value: int) -> float:
        return float((value / total) * 100.0) if total else 0.0

    return {
        "total": total,
        "fire": fire,
        "no_fire": no_fire,
        "fire_percent": pct(fire),
        "no_fire_percent": pct(no_fire),
        "bins": {name: int(counts["bins"].get(name, 0)) for name in BIN_ORDER},
        "bin_percentages": {name: pct(int(counts["bins"].get(name, 0))) for name in BIN_ORDER},
        "active_pixels": active_pixels,
        "total_pixels": total_pixels,
        "active_pixel_percent": float((active_pixels / total_pixels) * 100.0) if total_pixels else 0.0,
    }


def diagnose_balance(fire_percent: float) -> str:
    if fire_percent < 20.0:
        return "Dataset appears background-heavy."
    if fire_percent > 80.0:
        return "Dataset appears fire-heavy."
    return "Dataset is moderately balanced."


def _format_count_line(label: str, value: int, percent: float) -> str:
    return f"{label:<15} {value:>10d}  ({percent:6.2f}%)"


def print_summary(summary: Mapping[str, Any], per_fire: bool = False) -> None:
    counts = summary["counts"]
    print("=" * 60)
    print("Patch Fire Balance Check")
    print("=" * 60)
    print(f"Dataset root: {summary['dataset_root']}")
    print(f"Sample pattern: {summary['sample_pattern']}")
    print(f"Sample index: {summary['sample_index']}")
    print(f"Split: {summary['split']}")
    print(f"Active threshold: mask > {summary['fire_threshold']}")
    cmp = ">" if float(summary["active_fraction_threshold"]) <= 0.0 else ">="
    print(f"Fire patch definition: active_fraction {cmp} {summary['active_fraction_threshold']}")
    print()
    print(f"Total patchified samples: {counts['total']}")
    print()
    print(_format_count_line("Fire samples:", counts["fire"], counts["fire_percent"]))
    print(_format_count_line("No-fire samples:", counts["no_fire"], counts["no_fire_percent"]))
    print(f"Active pixels across all sampled patches: {counts['active_pixels']} / {counts['total_pixels']} ({counts['active_pixel_percent']:.4f}%)")
    print()
    print("Active-fraction bins:")
    for name in BIN_ORDER:
        print(_format_count_line(f"  {name}", counts["bins"][name], counts["bin_percentages"][name]))

    if per_fire:
        print()
        print("Per-fire breakdown:")
        for fire_name, fire_counts in summary["per_fire"].items():
            print(f"  {fire_name}:")
            print(f"    total: {fire_counts['total']}")
            print(f"    fire: {fire_counts['fire']} ({fire_counts['fire_percent']:.2f}%)")
            print(f"    no_fire: {fire_counts['no_fire']} ({fire_counts['no_fire_percent']:.2f}%)")
            print(f"    active pixels: {fire_counts['active_pixels']} / {fire_counts['total_pixels']} ({fire_counts['active_pixel_percent']:.4f}%)")
    print()
    print(summary["diagnosis"])


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check fire/no-fire balance in processed patchified samples.")
    parser.add_argument("--dataset-root", default="/scratch/mhabibp/cawfe_datasets/cawfe_engineered_v1")
    parser.add_argument("--sample-pattern", choices=SAMPLE_PATTERNS, default="consecutive5_h10")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--fire-threshold", type=float, default=0.5)
    parser.add_argument("--active-fraction-threshold", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--per-fire", action="store_true")
    parser.add_argument("--save-json")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    summary = check_balance(
        dataset_root=args.dataset_root,
        sample_pattern=args.sample_pattern,
        split=args.split,
        fire_threshold=args.fire_threshold,
        active_fraction_threshold=args.active_fraction_threshold,
        max_samples=args.max_samples,
        verbose=args.verbose,
    )
    print_summary(summary, per_fire=args.per_fire)
    if args.save_json:
        output_path = Path(args.save_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved JSON summary: {output_path}")


if __name__ == "__main__":
    main()
