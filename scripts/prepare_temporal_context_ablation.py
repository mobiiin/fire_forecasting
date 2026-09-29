#!/usr/bin/env python3
"""Prepare a timestamp-controlled FLARE temporal-context ablation.

The processed dataset does not store timestamps. For this experiment, the user
confirmed that the numeric suffix in ``tensorNNNN`` is an elapsed-minute
coordinate. This script treats that suffix as authoritative, builds one shared
set of forecast reference cases, fits train-only channel-wise normalization for
each temporal mode, writes resolved configs, and performs preflight checks. It
never reads test samples or trains a network.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from scripts.compute_processed_dataset_normalization import _EngineeredFrameCache
from src.config import load_config
from src.data.dataset import create_dataloaders
from src.data.processed_sample_dataset import ProcessedTemporalPatchDataset
from src.evaluation.validation_subset import ensure_screening_validation_indices
from src.models.model_factory import build_model_from_config
from src.training.batch_utils import unpack_batch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs/final_training/cawfe_latte_baseline_full.yaml"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts/temporal_context_ablation"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results/temporal_context_ablation"
MODE_ORDER = ("single", "sparse5", "dense5")
MODE_T = {"single": 1, "sparse5": 5, "dense5": 5}
FIXED_OFFSETS_MINUTES = {
    "single": [0],
    "sparse5": [-40, -30, -20, -10, 0],
}
SEEDS = (42, 123, 2026)
TARGET_HORIZON_MINUTES = 10
CADENCE_SOURCE = "user-confirmed numeric suffix of source_raw_file tensorNNNN; one suffix increment equals one minute"
TENSOR_SUFFIX = re.compile(r"tensor(\d+)$", re.IGNORECASE)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_builtin(item) for item in value]
    return value


def _atomic_text(path: Path, text: str, *, force: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    if path.is_file():
        if path.read_bytes() == encoded:
            return "reused"
        if not force:
            raise FileExistsError(
                f"Refusing to replace a non-matching preparation artifact: {path}. "
                "Inspect it or rerun with --force."
            )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return "written"


def _atomic_json(path: Path, payload: Any, *, force: bool) -> str:
    return _atomic_text(path, json.dumps(_json_builtin(payload), indent=2, sort_keys=True) + "\n", force=force)


def _atomic_yaml(path: Path, payload: Mapping[str, Any], *, force: bool) -> str:
    return _atomic_text(path, yaml.safe_dump(_json_builtin(dict(payload)), sort_keys=False), force=force)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, force: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_builtin(dict(row)), sort_keys=True, separators=(",", ":")) + "\n")
    if path.is_file() and sha256_file(path) == sha256_file(temporary):
        temporary.unlink()
        return "reused"
    if path.exists() and not force:
        temporary.unlink()
        raise FileExistsError(
            f"Refusing to replace a non-matching preparation artifact: {path}. "
            "Inspect it or rerun with --force."
        )
    temporary.replace(path)
    return "written"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tensor_minute(source_raw_file: str) -> int:
    match = TENSOR_SUFFIX.fullmatch(Path(source_raw_file).stem)
    if match is None:
        raise ValueError(f"Cannot extract tensorNNNN minute coordinate from {source_raw_file!r}.")
    return int(match.group(1))


def load_frame_timeline(dataset_root: Path, fire_name: str) -> list[dict[str, Any]]:
    path = dataset_root / "fires" / fire_name / "frame_manifest.jsonl"
    rows = _load_jsonl(path)
    if not rows:
        raise ValueError(f"Empty frame manifest: {path}")
    result: list[dict[str, Any]] = []
    seen_local: set[int] = set()
    seen_minutes: set[int] = set()
    for source in rows:
        local_index = int(source["local_index"])
        minute = _tensor_minute(str(source["source_raw_file"]))
        if local_index in seen_local:
            raise ValueError(f"Duplicate local frame index for {fire_name}: {local_index}")
        if minute in seen_minutes:
            raise ValueError(f"Duplicate tensor-minute coordinate for {fire_name}: {minute}")
        seen_local.add(local_index)
        seen_minutes.add(minute)
        result.append({**source, "local_index": local_index, "time_minutes": minute})
    result.sort(key=lambda row: int(row["local_index"]))
    if [int(row["local_index"]) for row in result] != list(range(len(result))):
        raise ValueError(f"Local frame indices are not contiguous for {fire_name}.")
    if any(int(result[index]["time_minutes"]) <= int(result[index - 1]["time_minutes"]) for index in range(1, len(result))):
        raise ValueError(f"tensorNNNN minute coordinates are not strictly increasing for {fire_name}.")
    return result


def _split_lists(base_config: Mapping[str, Any], dataset_manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    manual = base_config.get("manual_fire_split", {})
    if not isinstance(manual, Mapping) or not bool(manual.get("enabled", False)):
        raise ValueError("The canonical baseline must use an enabled manual_fire_split.")
    dataset_splits = dataset_manifest.get("splits", {})
    result: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        configured = [str(value) for value in manual.get(f"{split}_fires", [])]
        recorded = [str(value) for value in dataset_splits.get(split, dataset_splits.get(f"{split}_fires", []))]
        if configured != recorded:
            raise ValueError(
                f"Canonical config and processed dataset disagree on {split} fires. "
                f"config={configured}, dataset={recorded}"
            )
        result[split] = configured
    if set(result["train"]) & set(result["val"]):
        raise ValueError("Train and validation fires overlap.")
    return result


def _patches_by_fire(dataset_root: Path, allowed_fires: set[str]) -> dict[str, list[dict[str, Any]]]:
    patch_path = dataset_root / "indices/patches/patches_64_stride60_border.jsonl"
    patches = _load_jsonl(patch_path)
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in allowed_fires}
    for patch in patches:
        fire_name = str(patch["fire_name"])
        if fire_name in grouped:
            grouped[fire_name].append(patch)
    missing = sorted(name for name, rows in grouped.items() if not rows)
    if missing:
        raise ValueError(f"No patch definitions found for fires: {missing}")
    for rows in grouped.values():
        rows.sort(key=lambda row: (int(row["y0"]), int(row["x0"]), str(row.get("patch_id", ""))))
    return grouped


def _common_id(fire_name: str, patch: Mapping[str, Any], current_minute: int) -> str:
    return (
        f"{fire_name}_patch_y{int(patch['y0']):03d}_x{int(patch['x0']):03d}_"
        f"tminute{int(current_minute):06d}"
    )


def build_common_records(
    dataset_root: Path,
    split_fires: Mapping[str, Sequence[str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Return shared train/validation cases using exact tensor-minute matching."""

    allowed = set(split_fires["train"]) | set(split_fires["val"])
    patches = _patches_by_fire(dataset_root, allowed)
    records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    exclusion: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}
    timing: dict[str, Any] = {
        "fires": {},
        "dense_span_minutes": {"train": Counter(), "val": Counter()},
        "dense_offset_patterns_minutes": {"train": Counter(), "val": Counter()},
    }

    for split in ("train", "val"):
        for fire_name in split_fires[split]:
            timeline = load_frame_timeline(dataset_root, fire_name)
            by_minute = {int(row["time_minutes"]): row for row in timeline}
            minute_steps = [
                int(timeline[index]["time_minutes"]) - int(timeline[index - 1]["time_minutes"])
                for index in range(1, len(timeline))
            ]
            accepted_references = 0
            for position, current_row in enumerate(timeline):
                if position < 4:
                    exclusion[split]["insufficient_dense_history"] += 1
                    continue
                current_minute = int(current_row["time_minutes"])
                sparse_rows = [by_minute.get(current_minute + offset) for offset in FIXED_OFFSETS_MINUTES["sparse5"]]
                if any(row is None for row in sparse_rows):
                    exclusion[split]["missing_exact_sparse_time"] += 1
                    continue
                target_row = by_minute.get(current_minute + TARGET_HORIZON_MINUTES)
                if target_row is None:
                    exclusion[split]["missing_exact_target_time"] += 1
                    continue
                dense_rows = timeline[position - 4 : position + 1]
                dense_span = current_minute - int(dense_rows[0]["time_minutes"])
                dense_offsets = tuple(int(row["time_minutes"]) - current_minute for row in dense_rows)
                target_index = int(target_row["local_index"])
                current_index = int(current_row["local_index"])
                target_rel = (
                    Path("targets") / f"h{TARGET_HORIZON_MINUTES}" / fire_name /
                    f"target_current_{current_index:06d}_future_{target_index:06d}.npz"
                )
                if not (dataset_root / target_rel).is_file():
                    exclusion[split]["missing_exact_target_artifact"] += 1
                    continue
                input_rows = {
                    "single": [current_row],
                    "sparse5": [row for row in sparse_rows if row is not None],
                    "dense5": dense_rows,
                }
                if any(
                    not (dataset_root / "fires" / fire_name / "frames" / f"frame_{int(row['local_index']):06d}.npz").is_file()
                    for mode_rows in input_rows.values()
                    for row in mode_rows
                ):
                    exclusion[split]["missing_input_frame_artifact"] += 1
                    continue
                accepted_references += 1
                timing["dense_span_minutes"][split][dense_span] += len(patches[fire_name])
                timing["dense_offset_patterns_minutes"][split][dense_offsets] += len(patches[fire_name])
                for patch in patches[fire_name]:
                    sample_id = _common_id(fire_name, patch, current_minute)
                    records[split].append(
                        {
                            "sample_id": sample_id,
                            "shared_reference_id": sample_id,
                            "split": split,
                            "fire_name": fire_name,
                            "patch_id": str(patch["patch_id"]),
                            "patch": {key: int(patch[key]) for key in ("y0", "x0", "height", "width")},
                            "current_index": current_index,
                            "current_time_minutes": current_minute,
                            "target_index": target_index,
                            "target_time_minutes": int(target_row["time_minutes"]),
                            "horizon": TARGET_HORIZON_MINUTES,
                            "horizon_minutes": TARGET_HORIZON_MINUTES,
                            "target_path": str(target_rel),
                            "mode_input_indices": {
                                mode: [int(row["local_index"]) for row in mode_rows]
                                for mode, mode_rows in input_rows.items()
                            },
                            "mode_input_times_minutes": {
                                mode: [int(row["time_minutes"]) for row in mode_rows]
                                for mode, mode_rows in input_rows.items()
                            },
                        }
                    )
            timing["fires"][fire_name] = {
                "split": split,
                "frame_count": len(timeline),
                "first_tensor_minute": int(timeline[0]["time_minutes"]),
                "last_tensor_minute": int(timeline[-1]["time_minutes"]),
                "native_step_minutes_counts": dict(sorted(Counter(minute_steps).items())),
                "accepted_reference_time_count": accepted_references,
                "patch_count": len(patches[fire_name]),
                "accepted_patch_sample_count": accepted_references * len(patches[fire_name]),
            }

    for split in ("train", "val"):
        if not records[split]:
            raise ValueError(f"No common {split} samples satisfy the exact time constraints.")
        ids = [str(row["sample_id"]) for row in records[split]]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Duplicate common sample IDs in {split}.")
    timing["excluded_reference_times"] = {split: dict(counter) for split, counter in exclusion.items()}
    timing["dense_span_minutes"] = {
        split: {str(key): int(value) for key, value in sorted(counter.items())}
        for split, counter in timing["dense_span_minutes"].items()
    }
    timing["dense_offset_patterns_minutes"] = {
        split: {json.dumps(list(key), separators=(",", ":")): int(value) for key, value in sorted(counter.items())}
        for split, counter in timing["dense_offset_patterns_minutes"].items()
    }
    return records, timing


def mode_record(common: Mapping[str, Any], mode: str) -> dict[str, Any]:
    times = [int(value) for value in common["mode_input_times_minutes"][mode]]
    latest = int(common["current_time_minutes"])
    return {
        "sample_id": str(common["sample_id"]),
        "shared_reference_id": str(common["shared_reference_id"]),
        "split": str(common["split"]),
        "fire_name": str(common["fire_name"]),
        "pattern": f"temporal_context_{mode}",
        "patch_id": str(common["patch_id"]),
        "patch": dict(common["patch"]),
        "input_indices": [int(value) for value in common["mode_input_indices"][mode]],
        "input_times_minutes": times,
        "temporal_offsets_minutes": [value - latest for value in times],
        "current_index": int(common["current_index"]),
        "current_time_minutes": latest,
        "target_index": int(common["target_index"]),
        "target_time_minutes": int(common["target_time_minutes"]),
        "horizon": TARGET_HORIZON_MINUTES,
        "horizon_minutes": TARGET_HORIZON_MINUTES,
        "target_path": str(common["target_path"]),
        "time_coordinate_source": "source_raw_file tensorNNNN suffix",
    }


def write_indices(
    artifact_root: Path,
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    force: bool,
) -> dict[str, Any]:
    paths: dict[str, Any] = {"common": {}, "modes": {}}
    for split in ("train", "val"):
        path = artifact_root / f"common_{split}_index.jsonl"
        _atomic_jsonl(path, records[split], force=force)
        paths["common"][split] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    combined = [*records["train"], *records["val"]]
    for mode in MODE_ORDER:
        path = artifact_root / mode / "sample_index.jsonl"
        _atomic_jsonl(path, (mode_record(row, mode) for row in combined), force=force)
        paths["modes"][mode] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return paths


def _normalization_identity(
    *,
    mode: str,
    sample_index: Path,
    base_config: Path,
    dataset_root: Path,
    train_count: int,
) -> dict[str, Any]:
    return {
        "normalization_version": "temporal_context_ablation_train_only_channelwise_v1",
        "mode": mode,
        "fit_split": "train",
        "fit_fire_scope": "canonical_baseline_train_fires_only",
        "temporal_sample_index_path": str(sample_index.resolve()),
        "temporal_sample_index_sha256": sha256_file(sample_index),
        "num_train_samples_used": int(train_count),
        "base_config_path": str(base_config.resolve()),
        "base_config_sha256": sha256_file(base_config),
        "dataset_root": str(dataset_root.resolve()),
        "dataset_manifest_sha256": sha256_file(dataset_root / "dataset_manifest.json"),
        "channel_manifest_sha256": sha256_file(dataset_root / "channel_manifest.json"),
        "input_channels": 129,
        "statistics_layout": "per_channel_C; broadcast identically over temporal slots",
        "time_coordinate_source": CADENCE_SOURCE,
    }


def _validate_normalization(json_path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    identity = payload.get("identity")
    if identity != dict(expected):
        raise ValueError(f"Normalization provenance mismatch in {json_path}.")
    if payload.get("fit_split") != "train":
        raise ValueError(f"Normalization is not explicitly marked train-only: {json_path}")
    if int(payload.get("input_channels", -1)) != 129:
        raise ValueError(f"Normalization does not describe 129 channels: {json_path}")
    npz_path = Path(str(payload["npz_path"]))
    if not npz_path.is_file():
        raise FileNotFoundError(f"Normalization NPZ is missing: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        for key in ("mean", "std", "min", "max"):
            values = np.asarray(archive[key])
            if values.shape != (129,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {key} in {npz_path}: shape={values.shape}")
        if bool((np.asarray(archive["std"]) <= 0).any()):
            raise ValueError(f"Normalization standard deviation is not positive: {npz_path}")
    if sha256_file(npz_path) != payload.get("npz_sha256"):
        raise ValueError(f"Normalization hash mismatch: {npz_path}")
    return payload


def compute_or_validate_normalization(
    *,
    mode: str,
    sample_index: Path,
    base_config: Path,
    dataset_root: Path,
    output_dir: Path,
    train_count: int,
    cache: _EngineeredFrameCache,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "normalization.json"
    npz_path = output_dir / "normalization.npz"
    identity = _normalization_identity(
        mode=mode,
        sample_index=sample_index,
        base_config=base_config,
        dataset_root=dataset_root,
        train_count=train_count,
    )
    if json_path.is_file() and not force:
        payload = _validate_normalization(json_path, identity)
        print(f"Reusing validated normalization: {json_path}")
        return payload
    if (json_path.exists() or npz_path.exists()) and not force:
        raise FileExistsError(f"Incomplete normalization artifact in {output_dir}; inspect it or use --force.")

    channel_manifest = json.loads((dataset_root / "channel_manifest.json").read_text(encoding="utf-8"))
    channel_entries = channel_manifest.get("channels", [])
    channel_names = [str(entry.get("name", entry)) if isinstance(entry, Mapping) else str(entry) for entry in channel_entries]
    channels = 129
    total_pixels = 0
    sample_count = 0
    sum_ = np.zeros(channels, dtype=np.float64)
    sumsq = np.zeros(channels, dtype=np.float64)
    min_ = np.full(channels, np.inf, dtype=np.float64)
    max_ = np.full(channels, -np.inf, dtype=np.float64)
    with sample_index.open("r", encoding="utf-8") as count_handle:
        total_lines = sum(1 for line in count_handle if line.strip())
    with sample_index.open("r", encoding="utf-8") as handle:
        iterator = tqdm(handle, total=total_lines, desc=f"Normalization {mode}", unit="sample")
        for line in iterator:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != "train":
                continue
            sample_count += 1
            patch = record["patch"]
            y0, x0, height, width = (int(patch[key]) for key in ("y0", "x0", "height", "width"))
            for frame_index in record["input_indices"]:
                frame = cache.get(dataset_root, str(record["fire_name"]), int(frame_index))
                if frame.shape[0] != channels:
                    raise ValueError(f"Expected 129 channels, got {frame.shape} for {record['sample_id']}")
                values = np.asarray(frame[:, y0 : y0 + height, x0 : x0 + width], dtype=np.float64)
                if values.shape != (channels, height, width) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid normalization input for {record['sample_id']}: {values.shape}")
                flattened = values.reshape(channels, -1)
                total_pixels += flattened.shape[1]
                sum_ += flattened.sum(axis=1)
                sumsq += np.square(flattened).sum(axis=1)
                min_ = np.minimum(min_, flattened.min(axis=1))
                max_ = np.maximum(max_, flattened.max(axis=1))
    if sample_count != train_count:
        raise RuntimeError(f"Normalization read {sample_count} train samples, expected {train_count}.")
    mean = sum_ / float(total_pixels)
    std = np.sqrt(np.maximum(sumsq / float(total_pixels) - np.square(mean), 0.0))
    std = np.maximum(std, 1.0e-6)
    arrays = {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "min": min_.astype(np.float32),
        "max": max_.astype(np.float32),
        "count": np.asarray(total_pixels, dtype=np.int64),
        "channel_indices": np.arange(channels, dtype=np.int64),
        "channel_names": np.asarray(channel_names, dtype="U"),
    }
    temporary_npz = npz_path.with_name(npz_path.name + ".tmp")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary_npz.replace(npz_path)
    payload = {
        "normalization_version": identity["normalization_version"],
        "fit_split": "train",
        "num_samples_used": int(sample_count),
        "input_channels": channels,
        "temporal_sample_index_path": str(sample_index.resolve()),
        "temporal_sample_index_sha256": sha256_file(sample_index),
        "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
        "channel_manifest_sha256": identity["channel_manifest_sha256"],
        "identity": identity,
        "npz_path": str(npz_path.resolve()),
        "npz_sha256": sha256_file(npz_path),
        "pixel_count": int(total_pixels),
        "frame_cache": {
            "max_bytes": int(cache.max_bytes),
            "hits": int(cache.hits),
            "misses": int(cache.misses),
            "bytes_used": int(cache.bytes_used),
        },
    }
    _atomic_json(json_path, payload, force=True)
    return _validate_normalization(json_path, identity)


def _set_recursive(config: Any, key: str, value: Any) -> None:
    if isinstance(config, dict):
        if key in config:
            config[key] = deepcopy(value)
        for nested in config.values():
            _set_recursive(nested, key, value)
    elif isinstance(config, list):
        for nested in config:
            _set_recursive(nested, key, value)


def _assert_original_baseline(config: Mapping[str, Any]) -> None:
    if config.get("model", {}).get("architecture") != "cawfe_latte":
        raise ValueError("Canonical baseline architecture is not cawfe_latte.")
    cawfe = config.get("cawfe_latte", {})
    required = {
        "cawfe_latte.ablation.name": cawfe.get("ablation", {}).get("name"),
        "cawfe_latte.post_fusion_backbone.type": cawfe.get("post_fusion_backbone", {}).get("type"),
        "cawfe_latte.temporal_pooling.type": cawfe.get("temporal_pooling", {}).get("type"),
        "cawfe_latte.decoder.type": cawfe.get("decoder", {}).get("type"),
    }
    expected = {
        "cawfe_latte.ablation.name": "baseline",
        "cawfe_latte.post_fusion_backbone.type": "baseline_cnn",
        "cawfe_latte.temporal_pooling.type": "baseline",
        "cawfe_latte.decoder.type": "shared",
    }
    if required != expected:
        raise ValueError(f"Resolved config is not the preserved original baseline: {required}")
    for name in ("domain_adversarial", "fire_mmd", "patch_fire_head", "mask_guided_regression", "regression_moe", "supervised_contrastive", "physical_state_aux"):
        if bool(cawfe.get(name, {}).get("enabled", False)):
            raise ValueError(f"Original baseline unexpectedly enables cawfe_latte.{name}.")


def build_mode_config(
    base_config: Mapping[str, Any],
    *,
    mode: str,
    sample_index: Path,
    normalization_json: Path,
    normalization_npz: Path,
    artifact_root: Path,
    result_root: Path,
    sample_counts: Mapping[str, int],
    index_sha256: str,
) -> dict[str, Any]:
    config = deepcopy(dict(base_config))
    for key in list(config):
        if str(key).startswith("_") or key in {"config_path", "base_config"}:
            config.pop(key, None)
    _set_recursive(config, "input_sequence_length", MODE_T[mode])
    _set_recursive(config, "prediction_horizon", TARGET_HORIZON_MINUTES)
    config["input_sequence_length"] = MODE_T[mode]
    config["prediction_horizon"] = TARGET_HORIZON_MINUTES
    config["experiment"] = {
        "name": f"flare_temporal_context_{mode}",
        "description": "Ten-epoch controlled temporal-input ablation using the preserved original FLARE/CAWFE-Latte baseline.",
    }
    dataloader = dict(config.get("dataloader", {}))
    dataloader.update(
        {
            "source": "processed_full_frames",
            "sample_pattern": f"temporal_context_{mode}",
            "sample_index_path": str(sample_index.resolve()),
            "include_test_split": False,
            "target_horizon": TARGET_HORIZON_MINUTES,
            "single_frame_mode": "as_is",
            "repeat_to_length": None,
            "normalize_inputs": True,
            "return_metadata": True,
        }
    )
    config["dataloader"] = dataloader
    config["return_metadata"] = True
    normalization = dict(config.get("normalization", {}))
    normalization.update(
        {
            "enabled": True,
            "require_stats": True,
            "fit_split": "train",
            "allow_val_test_fit": False,
            "apply_to_splits": ["train", "val"],
            "mode": "per_channel_common_reference_train_population",
            "sample_pattern": f"temporal_context_{mode}",
            "stats_path": str(normalization_json.resolve()),
            "path": str(normalization_json.resolve()),
            "npz_path": str(normalization_npz.resolve()),
        }
    )
    config["normalization"] = normalization
    training = dict(config.get("training", {}))
    training.update(
        {
            "max_epochs": 10,
            "epochs": 10,
            "max_train_batches_per_epoch": None,
            "max_train_batches": 7500,
            "run_external_test_after_training": False,
            "run_test_after_training": False,
            "overwrite_run": False,
        }
    )
    performance = dict(training.get("performance", {}))
    performance["max_train_batches_per_epoch"] = None
    performance["auto_batch_size"] = False
    performance["cudnn_benchmark"] = False
    training["performance"] = performance
    early = dict(training.get("early_stopping", {}))
    early.update(
        {
            "enabled": True,
            "monitor": "val_loss",
            "mode": "min",
            "patience": 8,
            "min_delta": 0.001,
            "start_epoch": 10,
            "checkpoint_best": True,
            "restore_best_weights": False,
            "save_latest_on_stop": True,
            "stop_on_nan": True,
        }
    )
    training["early_stopping"] = early
    validation = dict(training.get("validation", {}))
    screening = dict(validation.get("screening", {}))
    screening.update(
        {
            "enabled": True,
            "sampling": "stratified_fixed",
            "seed": 12345,
            "max_batches": 50,
            "index_path": str((artifact_root / mode / "screening_validation_indices.json").resolve()),
        }
    )
    validation["screening"] = screening
    full = dict(validation.get("full", {}))
    full.update(
        {
            "enabled": True,
            "checkpoint": "best",
            "run_once_after_training": True,
            "save_analysis_data": True,
            "qualitative_index_path": str((artifact_root / mode / "qualitative_validation_samples.json").resolve()),
            "qualitative_seed": 24680,
            "qualitative_samples_per_group": 4,
        }
    )
    validation["full"] = full
    training["validation"] = validation
    output = dict(training.get("output", {}))
    output.update(
        {
            "root_dir": str((result_root / mode).resolve()),
            "flat_run_layout": True,
            "update_architecture_latest_checkpoint": False,
            "save_best_checkpoint": True,
            "save_latest_checkpoint": True,
            "save_resolved_config": True,
            "save_original_config": True,
            "save_run_summary": True,
            "save_normalization_stats_copy": True,
        }
    )
    training["output"] = output
    training.pop("seed", None)
    config["training"] = training
    config.pop("seed", None)
    checkpoint = dict(config.get("checkpoint", {}))
    checkpoint["resume"] = False
    config["checkpoint"] = checkpoint
    evaluation = dict(config.get("evaluation", {}))
    evaluation["split"] = "val"
    evaluation["use_test_for_checkpointing"] = False
    config["evaluation"] = evaluation
    config["final_training"] = {
        "finalist": f"temporal_{mode}",
        "source_ablation": "baseline",
        "components": ["baseline"],
    }
    config["temporal_context_ablation"] = {
        "mode": mode,
        "T": MODE_T[mode],
        "fixed_temporal_offsets_minutes": FIXED_OFFSETS_MINUTES.get(mode),
        "dense_definition": "four immediately preceding available native states plus latest" if mode == "dense5" else None,
        "target_horizon_minutes": TARGET_HORIZON_MINUTES,
        "time_coordinate_source": CADENCE_SOURCE,
        "common_sample_index_path": str(sample_index.resolve()),
        "common_sample_index_sha256": index_sha256,
        "common_train_sample_count": int(sample_counts["train"]),
        "common_validation_sample_count": int(sample_counts["val"]),
        "test_split_constructed": False,
    }
    _assert_original_baseline(config)
    return config


def _validate_mode_config(config: Mapping[str, Any], mode: str) -> None:
    _assert_original_baseline(config)
    training = config["training"]
    problems: list[str] = []
    if int(training.get("max_epochs", -1)) != 10 or int(training.get("epochs", -1)) != 10:
        problems.append("training must be capped at exactly 10 epochs")
    if int(training.get("max_train_batches", -1)) != 7500:
        problems.append("training.max_train_batches must be 7500")
    if int(training.get("batch_size", -1)) != 8:
        problems.append("training.batch_size must be 8")
    if training.get("max_train_batches_per_epoch") not in (None, 0, "null"):
        problems.append("legacy per-epoch cap must be disabled")
    if bool(training.get("run_test_after_training")) or bool(training.get("run_external_test_after_training")):
        problems.append("test evaluation must remain disabled")
    if bool(config["dataloader"].get("include_test_split", True)):
        problems.append("test split construction must be disabled")
    if int(config.get("input_sequence_length", -1)) != MODE_T[mode]:
        problems.append("input sequence length mismatch")
    if config["normalization"].get("fit_split") != "train":
        problems.append("normalization must be train-only")
    if problems:
        raise ValueError(f"Unsafe prepared config for {mode}: " + "; ".join(problems))


def _parameter_counts(config: Mapping[str, Any]) -> dict[str, int]:
    model = build_model_from_config(config, input_channels=129)
    before = sum(parameter.numel() for parameter in model.parameters())
    trainable_before = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model.alignment._spatial_position(64, 64, device=torch.device("cpu"), dtype=torch.float32)
    after = sum(parameter.numel() for parameter in model.parameters())
    trainable_after = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "before_lazy_spatial_position": int(before),
        "trainable_before_lazy_spatial_position": int(trainable_before),
        "with_64x64_spatial_position": int(after),
        "trainable_with_64x64_spatial_position": int(trainable_after),
    }


def run_sanity_checks(
    configs: Mapping[str, Mapping[str, Any]],
    *,
    matched_sample_count: int,
) -> dict[str, Any]:
    datasets: dict[str, ProcessedTemporalPatchDataset] = {}
    batch_checks: dict[str, Any] = {}
    parameter_checks: dict[str, Any] = {}
    screening_indices: dict[str, list[int]] = {}
    for mode in MODE_ORDER:
        config = configs[mode]
        train_loader, val_loader, test_loader = create_dataloaders(config)
        if test_loader is not None:
            raise RuntimeError(f"{mode} unexpectedly constructed a test loader.")
        train_batch = next(iter(train_loader))
        x, y, extra = unpack_batch(train_batch)
        terrain = extra.get("terrain")
        expected = (int(x.shape[0]), MODE_T[mode], 129, int(x.shape[-2]), int(x.shape[-1]))
        if tuple(x.shape) != expected or y.shape[1] != 4 or terrain is None:
            raise RuntimeError(f"Unexpected {mode} batch shapes: x={tuple(x.shape)} y={tuple(y.shape)} terrain={None if terrain is None else tuple(terrain.shape)}")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not torch.isfinite(terrain).all():
            raise RuntimeError(f"Non-finite batch values for {mode}.")
        batch_checks[mode] = {
            "dynamic_shape": list(x.shape),
            "target_shape": list(y.shape),
            "terrain_shape": list(terrain.shape),
            "normalization_path": str(config["normalization"]["stats_path"]),
            "test_loader": None,
        }
        parameter_checks[mode] = _parameter_counts(config)
        artifact = ensure_screening_validation_indices(
            val_loader.dataset,
            config,
            requested_samples=min(len(val_loader.dataset), 50 * int(val_loader.batch_size)),
            seed=12345,
            output_path=config["training"]["validation"]["screening"]["index_path"],
        )
        screening_indices[mode] = [int(value) for value in artifact["selected_sample_indices"]]
        datasets[mode] = val_loader.dataset
        print(f"{mode}: dynamic={tuple(x.shape)} target={tuple(y.shape)} terrain={tuple(terrain.shape)} parameters={parameter_checks[mode]['with_64x64_spatial_position']}")
    if len({tuple(values) for values in screening_indices.values()}) != 1:
        raise RuntimeError("The three modes did not select identical screening-validation sample indices.")
    if len({values["with_64x64_spatial_position"] for values in parameter_checks.values()}) != 1:
        raise RuntimeError(f"Parameter counts differ across temporal modes: {parameter_checks}")

    length = len(datasets["single"])
    if any(len(dataset) != length for dataset in datasets.values()):
        raise RuntimeError("Validation dataset lengths differ across modes.")
    requested = min(max(1, int(matched_sample_count)), length)
    selected = sorted(random.Random(24680).sample(range(length), requested))
    matched: list[dict[str, Any]] = []
    for index in selected:
        records = {mode: datasets[mode].records[index] for mode in MODE_ORDER}
        sample_ids = {str(record["sample_id"]) for record in records.values()}
        if len(sample_ids) != 1:
            raise RuntimeError(f"Matched sample IDs differ at validation index {index}: {sample_ids}")
        reference = records["single"]
        for mode, record in records.items():
            for key in ("fire_name", "patch", "current_index", "current_time_minutes", "target_index", "target_time_minutes", "target_path"):
                if record[key] != reference[key]:
                    raise RuntimeError(f"Matched {key} differs for mode={mode}, index={index}.")
        latest = int(reference["current_time_minutes"])
        if records["sparse5"]["input_times_minutes"] != [latest - 40, latest - 30, latest - 20, latest - 10, latest]:
            raise RuntimeError("Sparse timestamps are not exact ten-minute samples.")
        if int(reference["target_time_minutes"]) != latest + 10:
            raise RuntimeError("Target is not exactly ten minutes after the latest input.")
        loaded = {mode: datasets[mode][index] for mode in MODE_ORDER}
        targets = [loaded[mode]["y"] for mode in MODE_ORDER]
        terrains = [loaded[mode]["terrain"] for mode in MODE_ORDER]
        if not all(torch.equal(targets[0], value) for value in targets[1:]):
            raise RuntimeError(f"Targets differ for matched validation sample {next(iter(sample_ids))}.")
        if not all(torch.equal(terrains[0], value) for value in terrains[1:]):
            raise RuntimeError(f"Terrain differs for matched validation sample {next(iter(sample_ids))}.")
        matched.append(
            {
                "dataset_index": index,
                "sample_id": next(iter(sample_ids)),
                "fire_name": str(reference["fire_name"]),
                "patch": dict(reference["patch"]),
                "latest_input_time_minutes": latest,
                "target_time_minutes": int(reference["target_time_minutes"]),
                "single_times_minutes": list(records["single"]["input_times_minutes"]),
                "sparse5_times_minutes": list(records["sparse5"]["input_times_minutes"]),
                "dense5_times_minutes": list(records["dense5"]["input_times_minutes"]),
                "dense5_span_minutes": latest - int(records["dense5"]["input_times_minutes"][0]),
                "targets_identical": True,
                "terrain_identical": True,
            }
        )
    return {
        "batch_checks": batch_checks,
        "parameter_counts": parameter_checks,
        "screening_selected_sample_indices_identical": True,
        "screening_selected_sample_count": len(screening_indices["single"]),
        "matched_validation_samples": matched,
    }


def _report_text(manifest: Mapping[str, Any]) -> str:
    counts = manifest["sample_counts"]
    timing = manifest["timing_audit"]
    lines = [
        "=" * 60,
        "TEMPORAL CONTEXT ABLATION PREPARATION REPORT",
        "=" * 60,
        "",
        f"Baseline config: {manifest['base_config']['path']}",
        f"Baseline config SHA-256: {manifest['base_config']['sha256']}",
        "Architecture: original FLARE / CAWFE-Latte baseline",
        "Post-fusion backbone: baseline_cnn",
        "Temporal pooling: baseline (last state)",
        "Decoder: shared",
        "Epoch cap: exactly 10",
        "Training batches per computational epoch: 7500",
        "Batch size: 8",
        "Checkpoint selection: minimum screening validation val_loss",
        "Held-out test split constructed/read: no",
        "",
        "TIME PROVENANCE",
        f"- {manifest['time_coordinate_source']}",
        "- target horizon: +10 exact tensor-minute units from the latest input",
        "- samples without every required exact sparse/target time are excluded",
        "",
        "SPLITS",
        f"Train fires ({len(manifest['train_fires'])}): {', '.join(manifest['train_fires'])}",
        f"Validation fires ({len(manifest['validation_fires'])}): {', '.join(manifest['validation_fires'])}",
        f"Common train patch samples: {counts['train']}",
        f"Common validation patch samples: {counts['val']}",
        "",
        "TEMPORAL DEFINITIONS",
        "Single: T=1; offsets [0] minutes",
        "Sparse5: T=5; offsets [-40, -30, -20, -10, 0] minutes",
        "Dense5: T=5; four immediately preceding available native CAWFE states plus latest",
        f"Dense5 train span distribution (minutes): {timing['dense_span_minutes']['train']}",
        f"Dense5 validation span distribution (minutes): {timing['dense_span_minutes']['val']}",
        f"Dense5 train offset-pattern distribution: {timing['dense_offset_patterns_minutes']['train']}",
        f"Dense5 validation offset-pattern distribution: {timing['dense_offset_patterns_minutes']['val']}",
        "",
        "DATASET TIMING AUDIT",
        f"- excluded train reference times: {timing['excluded_reference_times']['train']}",
        f"- excluded validation reference times: {timing['excluded_reference_times']['val']}",
    ]
    for fire_name, item in timing["fires"].items():
        lines.append(
            f"- {fire_name} ({item['split']}): native minute-step counts={item['native_step_minutes_counts']}; "
            f"retained reference times={item['accepted_reference_time_count']}; "
            f"retained patch samples={item['accepted_patch_sample_count']}"
        )
    lines.extend([
        "",
        "NORMALIZATION",
        "- per-channel C=129 statistics; independent of temporal slot",
        "- separately fitted for each mode because the training-frame populations differ",
        "- fit split: train only; validation and test are excluded",
        f"- feature-order identity: channel_manifest SHA-256 {manifest['channel_manifest_sha256']}",
    ])
    for mode in MODE_ORDER:
        norm = manifest["normalization"][mode]
        lines.append(f"- {mode}: {norm['json_path']} (SHA-256 {norm['json_sha256']})")
    lines.extend(["", "SANITY CHECKS"])
    sanity = manifest["sanity_checks"]
    for mode in MODE_ORDER:
        check = sanity["batch_checks"][mode]
        params = sanity["parameter_counts"][mode]["with_64x64_spatial_position"]
        lines.append(
            f"- {mode}: dynamic={check['dynamic_shape']} target={check['target_shape']} "
            f"terrain={check['terrain_shape']} parameters={params}"
        )
    lines.append(f"- identical stratified screening indices: {sanity['screening_selected_sample_indices_identical']}")
    lines.append(f"- matched validation samples checked: {len(sanity['matched_validation_samples'])}")
    for item in sanity["matched_validation_samples"]:
        lines.append(
            f"  {item['sample_id']} | fire={item['fire_name']} | latest={item['latest_input_time_minutes']} "
            f"target={item['target_time_minutes']} | sparse={item['sparse5_times_minutes']} "
            f"dense={item['dense5_times_minutes']} | terrain/target identical"
        )
    lines.extend(["", "CONFIGS"])
    for mode in MODE_ORDER:
        lines.append(f"- {mode}: {manifest['configs'][mode]['path']}")
    return "\n".join(lines) + "\n"


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    base_config_path = Path(args.base_config).expanduser().resolve()
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    result_root = Path(args.result_root).expanduser().resolve()
    base_config = load_config(base_config_path)
    _assert_original_baseline(base_config)
    dataset_root = Path(args.dataset_root or base_config["processed_dataset"]["root"]).expanduser().resolve()
    dataset_manifest = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    channel_manifest = json.loads((dataset_root / "channel_manifest.json").read_text(encoding="utf-8"))
    channel_count = int(channel_manifest.get("num_engineered_total_channels", channel_manifest.get("num_channels", len(channel_manifest.get("channels", [])))))
    if channel_count != 129:
        raise ValueError(f"Expected exactly 129 engineered channels, got {channel_count}.")
    split_fires = _split_lists(base_config, dataset_manifest)
    records, timing = build_common_records(dataset_root, split_fires)
    sample_counts = {split: len(records[split]) for split in ("train", "val")}
    index_info = write_indices(artifact_root, records, force=bool(args.force))

    if args.index_only:
        payload = {
            "schema_version": 1,
            "ready_for_training": False,
            "reason": "index_only requested; normalization/config/sanity preparation not run",
            "time_coordinate_source": CADENCE_SOURCE,
            "sample_counts": sample_counts,
            "train_fires": split_fires["train"],
            "validation_fires": split_fires["val"],
            "test_fires_not_used": split_fires["test"],
            "indices": index_info,
            "timing_audit": timing,
        }
        _atomic_json(artifact_root / "index_only_report.json", payload, force=bool(args.force))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return payload

    frame_cache = _EngineeredFrameCache(int(float(args.frame_cache_gb) * 1024**3))
    normalization: dict[str, Any] = {}
    for mode in MODE_ORDER:
        sample_index = Path(index_info["modes"][mode]["path"])
        payload = compute_or_validate_normalization(
            mode=mode,
            sample_index=sample_index,
            base_config=base_config_path,
            dataset_root=dataset_root,
            output_dir=artifact_root / mode,
            train_count=sample_counts["train"],
            cache=frame_cache,
            force=bool(args.force),
        )
        normalization[mode] = {
            "json_path": str((artifact_root / mode / "normalization.json").resolve()),
            "json_sha256": sha256_file(artifact_root / mode / "normalization.json"),
            "npz_path": str(Path(payload["npz_path"]).resolve()),
            "npz_sha256": payload["npz_sha256"],
            "fit_split": "train",
            "num_train_samples_used": sample_counts["train"],
            "pixel_count": int(payload["pixel_count"]),
        }

    configs: dict[str, dict[str, Any]] = {}
    config_info: dict[str, Any] = {}
    for mode in MODE_ORDER:
        sample_index = Path(index_info["modes"][mode]["path"])
        config = build_mode_config(
            base_config,
            mode=mode,
            sample_index=sample_index,
            normalization_json=artifact_root / mode / "normalization.json",
            normalization_npz=artifact_root / mode / "normalization.npz",
            artifact_root=artifact_root,
            result_root=result_root,
            sample_counts=sample_counts,
            index_sha256=index_info["modes"][mode]["sha256"],
        )
        _validate_mode_config(config, mode)
        config_path = artifact_root / mode / "config_resolved.yaml"
        _atomic_yaml(config_path, config, force=bool(args.force))
        loaded = load_config(config_path)
        _validate_mode_config(loaded, mode)
        configs[mode] = loaded
        config_info[mode] = {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)}

    sanity = run_sanity_checks(configs, matched_sample_count=int(args.sanity_samples))
    manifest = {
        "schema_version": 1,
        "ready_for_training": True,
        "base_config": {"path": str(base_config_path), "sha256": sha256_file(base_config_path)},
        "dataset_root": str(dataset_root),
        "dataset_manifest_sha256": sha256_file(dataset_root / "dataset_manifest.json"),
        "channel_manifest_sha256": sha256_file(dataset_root / "channel_manifest.json"),
        "input_channels": 129,
        "time_coordinate_source": CADENCE_SOURCE,
        "target_horizon_minutes": TARGET_HORIZON_MINUTES,
        "epoch_cap": 10,
        "seeds": list(SEEDS),
        "train_fires": split_fires["train"],
        "validation_fires": split_fires["val"],
        "test_fires_not_used": split_fires["test"],
        "sample_counts": sample_counts,
        "indices": index_info,
        "normalization": normalization,
        "configs": config_info,
        "timing_audit": timing,
        "sanity_checks": sanity,
    }
    _atomic_json(artifact_root / "preparation_manifest.json", manifest, force=bool(args.force))
    report_path = result_root / "preparation_report.txt"
    _atomic_text(report_path, _report_text(manifest), force=bool(args.force))
    print(_report_text(manifest), end="")
    print(f"Preparation manifest: {artifact_root / 'preparation_manifest.json'}")
    print(f"Preparation report: {report_path}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--frame-cache-gb", type=float, default=96.0)
    parser.add_argument("--sanity-samples", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Explicitly replace mismatching preparation artifacts.")
    parser.add_argument("--index-only", action="store_true", help="Build/audit shared indices only; outputs are not training-ready.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.frame_cache_gb < 0:
        raise ValueError("--frame-cache-gb must be non-negative.")
    if args.sanity_samples < 1:
        raise ValueError("--sanity-samples must be positive.")
    prepare(args)


if __name__ == "__main__":
    main()
