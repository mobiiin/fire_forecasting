"""Deterministic, target-stratified screening-validation subsets."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from src.evaluation.fire_activity import (
    ACTIVE_FRACTION_THRESHOLD,
    FIRE_MASK_THRESHOLD,
    active_fraction_bin_name,
    classify_patch_fire_state,
)


DEFAULT_SCREENING_INDEX_PATH = Path(
    "artifacts/ablations/cawfe_latte/shared_validation/screening_validation_indices.json"
)
DEFAULT_QUALITATIVE_INDEX_PATH = Path(
    "artifacts/full_training/shared_analysis/qualitative_validation_samples.json"
)
QUALITATIVE_SELECTION_SEED = 24680


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Target file not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        for key in ("fire_mask", "mask"):
            if key in archive.files:
                return np.asarray(archive[key])
        for key in ("y", "target"):
            if key in archive.files:
                value = np.asarray(archive[key])
                if value.ndim < 3 or value.shape[0] <= 2:
                    raise ValueError(f"Target array {key!r} cannot provide channel 2: {path} shape={value.shape}")
                return np.asarray(value[2])
        available = list(archive.files)
    raise KeyError(f"No target fire mask found in {path}; keys={available}")


def _record_target_path(dataset: Any, record: Mapping[str, Any]) -> Path:
    value = record.get("target_path", record.get("target_file"))
    if not value:
        raise KeyError(f"Validation record lacks target_path/target_file: {sorted(record)}")
    path = Path(str(value)).expanduser()
    root = Path(getattr(dataset, "root", ".")).expanduser().resolve()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _record_fire_name(record: Mapping[str, Any]) -> str:
    value = record.get("fire_name", record.get("fire"))
    if not value:
        raise KeyError(f"Validation record lacks fire_name/fire: {sorted(record)}")
    return str(value)


def _record_patch(record: Mapping[str, Any]) -> dict[str, int]:
    patch = record.get("patch")
    if not isinstance(patch, Mapping):
        raise KeyError(f"Validation record lacks patch metadata: {record.get('sample_id', '<unknown>')}")
    return {key: int(patch[key]) for key in ("y0", "x0", "height", "width")}


def classify_validation_records(
    dataset: Any,
    *,
    fire_threshold: float = FIRE_MASK_THRESHOLD,
    active_fraction_threshold: float = ACTIVE_FRACTION_THRESHOLD,
    progress_every: int = 1000,
    logger: Any = None,
) -> list[dict[str, Any]]:
    """Classify processed validation records without loading model inputs."""

    records = getattr(dataset, "records", None)
    if not isinstance(records, Sequence) or len(records) != len(dataset):
        raise TypeError("Stratified screening requires a validation dataset with one metadata record per sample.")
    classified: list[dict[str, Any]] = []
    cached_path: Path | None = None
    cached_mask: np.ndarray | None = None
    total = len(records)
    for index, record in enumerate(records):
        target_path = _record_target_path(dataset, record)
        if target_path != cached_path:
            cached_path = target_path
            cached_mask = _target_mask(target_path)
        assert cached_mask is not None
        state = classify_patch_fire_state(
            cached_mask,
            _record_patch(record),
            fire_threshold=fire_threshold,
            active_fraction_threshold=active_fraction_threshold,
        )
        classified.append(
            {
                "index": index,
                "sample_id": str(record.get("sample_id", index)),
                "fire_name": _record_fire_name(record),
                "class": "fire" if state["has_fire"] else "no_fire",
                "active_pixels": int(state["active_pixels"]),
                "active_fraction": float(state["active_fraction"]),
                "activity_bin": active_fraction_bin_name(float(state["active_fraction"])),
            }
        )
        completed = index + 1
        if logger is not None and progress_every > 0 and (completed == 1 or completed % progress_every == 0 or completed == total):
            logger.info("Classified validation target masks: %s/%s", completed, total)
    return classified


def _stable_seed(seed: int, *parts: str) -> int:
    material = ":".join((str(int(seed)), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _allocate_quotas(counts: Mapping[str, int], requested: int, *, seed: int, label: str) -> dict[str, int]:
    available = {name: int(count) for name, count in counts.items() if int(count) > 0}
    total_available = sum(available.values())
    requested = min(max(0, int(requested)), total_available)
    quotas = {name: 0 for name in available}
    if requested == 0:
        return quotas

    names = sorted(available)
    if requested >= len(names):
        for name in names:
            quotas[name] = 1
        remaining = requested - len(names)
    else:
        shuffled = list(names)
        random.Random(_stable_seed(seed, label, "coverage")).shuffle(shuffled)
        for name in shuffled[:requested]:
            quotas[name] = 1
        return quotas

    while remaining > 0:
        capacities = {name: available[name] - quotas[name] for name in names}
        capacity_total = sum(capacities.values())
        if capacity_total <= 0:
            break
        ideals = {name: remaining * capacities[name] / capacity_total for name in names}
        additions = {name: min(capacities[name], int(math.floor(ideals[name]))) for name in names}
        added = sum(additions.values())
        for name, value in additions.items():
            quotas[name] += value
        remaining -= added
        if remaining <= 0:
            break
        ranked = sorted(
            (name for name in names if quotas[name] < available[name]),
            key=lambda name: (
                -(ideals[name] - math.floor(ideals[name])),
                _stable_seed(seed, label, name, "remainder"),
            ),
        )
        for name in ranked[:remaining]:
            quotas[name] += 1
        remaining = 0
    if sum(quotas.values()) != requested:
        raise RuntimeError(f"Could not allocate {requested} {label} screening samples: {quotas}")
    return quotas


def select_stratified_indices(
    classified: Sequence[Mapping[str, Any]],
    requested_samples: int,
    *,
    seed: int,
) -> tuple[list[int], dict[str, Any]]:
    """Select deterministic natural-ratio fire/no-fire indices across fires."""

    total = len(classified)
    requested = min(int(requested_samples), total)
    if requested <= 0:
        raise ValueError("requested_samples must be positive.")
    by_class_fire: dict[str, dict[str, list[int]]] = {
        "fire": defaultdict(list),
        "no_fire": defaultdict(list),
    }
    for item in classified:
        label = str(item["class"])
        if label not in by_class_fire:
            raise ValueError(f"Unexpected validation class: {label!r}")
        by_class_fire[label][str(item["fire_name"])].append(int(item["index"]))
    class_counts = {label: sum(len(values) for values in per_fire.values()) for label, per_fire in by_class_fire.items()}
    if class_counts["fire"] == 0 or class_counts["no_fire"] == 0:
        raise RuntimeError(f"Representative screening requires both fire and no-fire validation samples: {class_counts}")

    fire_requested = int(round(requested * class_counts["fire"] / total))
    if requested >= 2:
        fire_requested = min(max(1, fire_requested), requested - 1)
    no_fire_requested = requested - fire_requested
    desired = {"fire": fire_requested, "no_fire": no_fire_requested}

    selected: list[int] = []
    selected_by_class_fire: dict[str, dict[str, int]] = {}
    for label in ("fire", "no_fire"):
        groups = by_class_fire[label]
        quotas = _allocate_quotas(
            {name: len(indices) for name, indices in groups.items()},
            desired[label],
            seed=seed,
            label=label,
        )
        selected_by_class_fire[label] = dict(sorted(quotas.items()))
        for fire_name, quota in sorted(quotas.items()):
            candidates = list(groups[fire_name])
            random.Random(_stable_seed(seed, label, fire_name, "samples")).shuffle(candidates)
            selected.extend(candidates[:quota])
    selected = sorted(selected)
    if len(selected) != requested or len(set(selected)) != requested:
        raise RuntimeError("Screening selection produced the wrong number of unique sample indices.")
    composition = {
        "total": requested,
        "fire": desired["fire"],
        "no_fire": desired["no_fire"],
        "fire_percent": 100.0 * desired["fire"] / requested,
        "no_fire_percent": 100.0 * desired["no_fire"] / requested,
        "per_class_per_fire": selected_by_class_fire,
    }
    return selected, composition


def _identity(dataset: Any, config: Mapping[str, Any], requested_samples: int, seed: int) -> dict[str, Any]:
    index_path = Path(getattr(dataset, "sample_index_path", "")).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Validation sample index is missing: {index_path}")
    dataloader = config.get("dataloader", {}) if isinstance(config.get("dataloader"), Mapping) else {}
    return {
        "dataset_root": str(Path(getattr(dataset, "root", ".")).expanduser().resolve()),
        "sample_index_path": str(index_path),
        "sample_index_sha256": _sha256_file(index_path),
        "split": str(getattr(dataset, "split", "val")),
        "sample_pattern": str(dataloader.get("sample_pattern", "")),
        "dataset_length": len(dataset),
        "patch_configuration": {
            "patch_size": config.get("patch_size"),
            "use_patches_for_eval": config.get("use_patches_for_eval"),
            "target_horizon": dataloader.get("target_horizon"),
        },
        "fire_threshold": FIRE_MASK_THRESHOLD,
        "active_fraction_threshold": ACTIVE_FRACTION_THRESHOLD,
        "seed": int(seed),
        "requested_samples": int(requested_samples),
    }


def ensure_screening_validation_indices(
    dataset: Any,
    config: Mapping[str, Any],
    *,
    requested_samples: int,
    seed: int,
    output_path: str | Path = DEFAULT_SCREENING_INDEX_PATH,
    logger: Any = None,
) -> dict[str, Any]:
    """Create or reuse the locked canonical screening-index artifact."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    identity = _identity(dataset, config, requested_samples, seed)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("identity") == identity:
                selected = existing.get("selected_sample_indices", [])
                if len(selected) != int(requested_samples) or len(set(selected)) != len(selected):
                    raise RuntimeError(f"Existing screening index artifact is invalid: {path}")
                if logger is not None:
                    logger.info("Reusing shared screening validation indices: %s", path)
                return existing
            if logger is not None:
                logger.warning("Screening validation identity changed; regenerating %s", path)

        classified = classify_validation_records(dataset, logger=logger)
        selected, screening_counts = select_stratified_indices(
            classified,
            requested_samples,
            seed=seed,
        )
        full_counts_raw = Counter(str(item["class"]) for item in classified)
        full_total = len(classified)
        full_counts = {
            "total": full_total,
            "fire": int(full_counts_raw["fire"]),
            "no_fire": int(full_counts_raw["no_fire"]),
            "fire_percent": 100.0 * full_counts_raw["fire"] / full_total,
            "no_fire_percent": 100.0 * full_counts_raw["no_fire"] / full_total,
        }
        by_index = {int(item["index"]): item for item in classified}
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "identity": identity,
            "full_validation_counts": full_counts,
            "screening_counts": screening_counts,
            "selected_sample_indices": selected,
            "selected_samples": [by_index[index] for index in selected],
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        if logger is not None:
            logger.info("Saved shared screening validation indices: %s", path)
        return payload


def _qualitative_group(activity_bin: str) -> str:
    name = str(activity_bin)
    if name == "no_fire":
        return "no_fire"
    if name in {"tiny_fire", "small_fire"}:
        return "weak_fire"
    if name in {"medium_fire", "large_fire"}:
        return name
    raise ValueError(f"Unknown canonical activity bin: {name!r}")


def select_qualitative_samples(
    classified: Sequence[Mapping[str, Any]],
    *,
    per_group: int = 4,
    seed: int = QUALITATIVE_SELECTION_SEED,
) -> list[dict[str, Any]]:
    """Select a deterministic, fire-diverse qualitative subset without model outputs."""

    requested = max(1, int(per_group))
    group_order = ("no_fire", "weak_fire", "medium_fire", "large_fire")
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        group: defaultdict(list) for group in group_order
    }
    for raw_item in classified:
        item = dict(raw_item)
        activity_bin = str(item.get("activity_bin", active_fraction_bin_name(float(item["active_fraction"]))))
        item["activity_bin"] = activity_bin
        group = _qualitative_group(activity_bin)
        grouped[group][str(item["fire_name"])].append(item)

    selected: list[dict[str, Any]] = []
    for group in group_order:
        candidates_by_fire = grouped[group]
        for fire_name, candidates in candidates_by_fire.items():
            candidates.sort(key=lambda item: _stable_seed(seed, group, fire_name, str(item["sample_id"])))
        fire_names = sorted(
            candidates_by_fire,
            key=lambda fire_name: _stable_seed(seed, group, fire_name, "fire_order"),
        )
        group_selected: list[dict[str, Any]] = []
        offset = 0
        while len(group_selected) < requested:
            added = False
            for fire_name in fire_names:
                candidates = candidates_by_fire[fire_name]
                if offset < len(candidates):
                    group_selected.append(candidates[offset])
                    added = True
                    if len(group_selected) >= requested:
                        break
            if not added:
                break
            offset += 1
        for item in group_selected:
            selected.append(
                {
                    "index": int(item["index"]),
                    "sample_id": str(item["sample_id"]),
                    "fire_name": str(item["fire_name"]),
                    "activity_bin": str(item["activity_bin"]),
                    "active_fraction": float(item["active_fraction"]),
                    "selection_group": group,
                }
            )
    if len({item["sample_id"] for item in selected}) != len(selected):
        raise RuntimeError("Qualitative selection produced duplicate sample IDs.")
    return selected


def ensure_qualitative_validation_samples(
    dataset: Any,
    config: Mapping[str, Any],
    *,
    output_path: str | Path = DEFAULT_QUALITATIVE_INDEX_PATH,
    per_group: int = 4,
    seed: int = QUALITATIVE_SELECTION_SEED,
    logger: Any = None,
) -> dict[str, Any]:
    """Create or reuse one dataset-identified qualitative validation subset."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    index_path = Path(getattr(dataset, "sample_index_path", "")).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Validation sample index is missing: {index_path}")
    identity = {
        "dataset_root": str(Path(getattr(dataset, "root", ".")).expanduser().resolve()),
        "sample_index_path": str(index_path),
        "sample_index_sha256": _sha256_file(index_path),
        "split": str(getattr(dataset, "split", "val")),
        "dataset_length": len(dataset),
        "selection_seed": int(seed),
        "samples_per_group": int(per_group),
        "groups": ["no_fire", "weak_fire", "medium_fire", "large_fire"],
        "canonical_activity_bins": ["no_fire", "tiny_fire", "small_fire", "medium_fire", "large_fire"],
    }
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("identity") == identity:
                samples = existing.get("selected_samples", [])
                if len({str(item["sample_id"]) for item in samples}) != len(samples):
                    raise RuntimeError(f"Existing qualitative subset contains duplicate IDs: {path}")
                if logger is not None:
                    logger.info("Reusing shared qualitative validation samples: %s", path)
                return existing
            if logger is not None:
                logger.warning("Qualitative validation identity changed; regenerating %s", path)

        classified = classify_validation_records(dataset, logger=logger)
        samples = select_qualitative_samples(classified, per_group=per_group, seed=seed)
        counts: dict[str, int] = Counter(str(item["selection_group"]) for item in samples)
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection_rule": "target_only_canonical_activity_bins_fire_diverse_fixed_seed",
            "identity": identity,
            "selected_sample_count": len(samples),
            "counts_by_selection_group": {group: int(counts.get(group, 0)) for group in identity["groups"]},
            "selected_sample_ids": [item["sample_id"] for item in samples],
            "selected_samples": samples,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        if logger is not None:
            logger.info("Saved shared qualitative validation samples: %s", path)
        return payload


__all__ = [
    "DEFAULT_QUALITATIVE_INDEX_PATH",
    "DEFAULT_SCREENING_INDEX_PATH",
    "classify_validation_records",
    "ensure_qualitative_validation_samples",
    "ensure_screening_validation_indices",
    "select_qualitative_samples",
    "select_stratified_indices",
]
