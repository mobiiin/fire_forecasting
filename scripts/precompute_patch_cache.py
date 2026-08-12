"""Precompute training-ready wildfire patch shards on scratch storage."""

from __future__ import annotations

import argparse
import atexit
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
from typing import Any, Mapping, Sequence

import numpy as np

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional .pt shards
	torch = None

from src.config import compute_file_sha256, compute_text_sha256, load_config
from src.data.cache import (
	MANIFEST_FILENAME,
	compute_cache_config_hash,
	compute_current_trim_metadata_hash,
	compute_dataset_index_hash,
	extract_temporal_trim_manifest,
	get_patch_cache_dir,
	load_cache_manifest,
	resolve_dataset_index_path,
	target_definition_version,
	temporal_target_offsets,
)
from src.data.dataset import MultiFirePatchSequenceDataset
from src.data.discovery import discover_multiple_datasets
from src.data.energy_release import resolve_energy_output_channel_names, resolve_energy_release_config
from src.data.patching import resolve_patching_config, resolve_split_patch_mode, resolve_split_patch_stride
from src.data.splits import build_sliding_patch_refs_for_split, manual_fire_holdout_splits, multi_fire_chronological_splits


def _get_pyplot():
	try:
		import matplotlib
		matplotlib.use("Agg", force=True)
		import matplotlib.pyplot as plt
	except ImportError:  # pragma: no cover - optional diagnostics
		return None
	return plt


def _get_section(config: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
	if not isinstance(config, Mapping):
		return {}
	section = config.get(name)
	return dict(section) if isinstance(section, Mapping) else {}


def _ensure_config_path(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
	resolved_path = Path(config_path).expanduser().resolve()
	config = dict(config)
	config["config_path"] = str(resolved_path)
	config["_config_path"] = str(resolved_path)
	return config


def _resolved_config_sha256(config: Mapping[str, Any]) -> str:
	encoded = json.dumps(config, sort_keys=True, default=str)
	return compute_text_sha256(encoded)


def _config_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = None if config_path_value in (None, "", "null") else Path(str(config_path_value)).expanduser().resolve()
	return {
		"config_path": str(config_path) if config_path is not None else None,
		"config_file_name": config_path.name if config_path is not None else None,
		"config_sha256": compute_file_sha256(config_path) if config_path is not None and config_path.exists() else None,
		"resolved_config_sha256": _resolved_config_sha256(config),
		"base_config_path": config.get("_base_config_path", config.get("base_config")),
		"base_config_sha256": config.get("_base_config_sha256"),
		"experiment_name": _get_section(config, "experiment").get("name"),
	}


def _acquire_cache_lock(cache_dir: Path, config: Mapping[str, Any], *, force: bool = False, ignore_stale_lock: bool = False) -> Path:
	lock_path = cache_dir / ".precompute_lock"
	lock_payload = {
		"created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
		"hostname": socket.gethostname(),
		"pid": os.getpid(),
		"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
		"cache_dir": str(cache_dir),
		"config_path": config.get("config_path", config.get("_config_path")),
	}
	if lock_path.exists() and not (force or ignore_stale_lock):
		try:
			existing = json.loads(lock_path.read_text(encoding="utf-8"))
		except Exception:
			existing = {"raw": lock_path.read_text(encoding="utf-8", errors="replace")}
		raise RuntimeError(
			f"Patch cache lock already exists: {lock_path}\n"
			f"Existing lock: {json.dumps(existing, sort_keys=True, default=str)}\n"
			"Use --ignore_stale_lock for an abandoned lock, or --force when intentionally taking ownership."
		)
	if not (force or ignore_stale_lock):
		try:
			fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
		except FileExistsError as exc:
			raise RuntimeError(f"Patch cache lock already exists: {lock_path}") from exc
		with os.fdopen(fd, "w", encoding="utf-8") as handle:
			json.dump(lock_payload, handle, indent=2, sort_keys=True)
			handle.write("\n")
		return lock_path
	temp_path = lock_path.with_name(f".{lock_path.name}.tmp.{os.getpid()}")
	temp_path.write_text(json.dumps(lock_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	os.replace(temp_path, lock_path)
	return lock_path


def _release_cache_lock(lock_path: Path) -> None:
	lock_path.unlink(missing_ok=True)


def _selected_splits(split: str) -> list[str]:
	split = str(split).lower()
	if split == "all":
		return ["train", "val", "test"]
	if split not in {"train", "val", "test"}:
		raise ValueError(f"split must be train, val, test, or all. Got {split!r}.")
	return [split]


def _atomic_write_text(path: Path, content: str) -> None:
	temporary_path = path.with_name(f".{path.name}.tmp")
	with temporary_path.open("w", encoding="utf-8") as handle:
		handle.write(content)
	temporary_path.replace(path)


def _shard_artifacts(split_dir: Path, shard_index: int, shard_format: str) -> tuple[Path, Path]:
	extension = ".pt" if str(shard_format).lower() == "pt" else ".npz"
	shard_path = split_dir / f"shard_{shard_index:06d}{extension}"
	metadata_path = split_dir / f"shard_{shard_index:06d}.metadata.jsonl"
	return shard_path, metadata_path


def _shard_index_from_path(path: Path) -> int:
	name = path.stem
	if not name.startswith("shard_"):
		raise ValueError(f"Unexpected shard filename: {path}")
	try:
		return int(name.split("_", 1)[1])
	except (IndexError, ValueError) as exc:
		raise ValueError(f"Unexpected shard filename: {path}") from exc


def _read_metadata_rows(path: Path) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line_number, line in enumerate(handle, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				value = json.loads(line)
			except json.JSONDecodeError as exc:
				raise ValueError(f"Invalid JSON in metadata file {path} on line {line_number}.") from exc
			if not isinstance(value, dict):
				raise ValueError(f"Metadata rows must be JSON objects. Got {type(value)!r} in {path}:{line_number}.")
			rows.append(value)
	return rows


def _read_shard_num_samples(shard_path: Path, shard_format: str) -> int:
	if str(shard_format).lower() == "pt":
		if torch is None:
			raise ImportError("PyTorch is required to resume cache.shard_format=pt shards.")
		shard = torch.load(shard_path, map_location="cpu")
		if "X" not in shard or "y" not in shard:
			raise ValueError(f"Shard is missing required X/y tensors: {shard_path}")
		return int(shard["X"].shape[0])
	with np.load(shard_path, allow_pickle=False) as shard:
		if "X" not in shard.files or "y" not in shard.files:
			raise ValueError(f"Shard is missing required X/y arrays: {shard_path}")
		return int(shard["X"].shape[0])


def _group_metadata_rows_by_shard(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
	grouped: dict[str, list[dict[str, Any]]] = {}
	for row in rows:
		shard_value = row.get("shard")
		if shard_value in (None, "", "null"):
			continue
		shard_name = Path(str(shard_value)).name
		grouped.setdefault(shard_name, []).append(dict(row))
	return grouped


def _normalized_shard_metadata_rows(
	rows: Sequence[Mapping[str, Any]],
	split: str,
	shard_name: str,
) -> list[dict[str, Any]]:
	normalized: list[dict[str, Any]] = []
	for local_index, row in enumerate(rows):
		item = dict(row)
		item["shard"] = f"{split}/{shard_name}"
		item["local_index"] = int(local_index)
		normalized.append(item)
	return normalized


def _parse_sample_id(sample_id: str) -> tuple[str, int | None, int | None, int | None]:
	parts = str(sample_id).split(":")
	if len(parts) < 5:
		return "", None, None, None
	fire_name = ":".join(parts[1:-3])
	try:
		sample_index = int(parts[-3])
	except ValueError:
		sample_index = None
	try:
		patch_y0 = int(parts[-2])
	except ValueError:
		patch_y0 = None
	try:
		patch_x0 = int(parts[-1])
	except ValueError:
		patch_x0 = None
	return fire_name, sample_index, patch_y0, patch_x0


def _sequence_metadata(sample_index: int, config: Mapping[str, Any]) -> dict[str, Any]:
	input_sequence_length = int(config["input_sequence_length"])
	prediction_horizon = int(config["prediction_horizon"])
	start_idx = int(sample_index)
	last_input_idx = start_idx + input_sequence_length - 1
	target_idx = last_input_idx + prediction_horizon
	offsets = temporal_target_offsets(config)
	return {
		"start_idx": start_idx,
		"input_indices": list(range(start_idx, start_idx + input_sequence_length)),
		"last_input_idx": int(last_input_idx),
		"target_idx": int(target_idx),
		"current_idx": int(last_input_idx),
		"future_idx": int(target_idx),
		"current_index": int(last_input_idx),
		"future_index": int(target_idx),
		"input_sequence_length": input_sequence_length,
		"prediction_horizon": prediction_horizon,
		"target_offset_from_start": int(offsets["target_offset_from_start"]),
		"target_offset_from_last_input": int(offsets["target_offset_from_last_input"]),
		"target_definition_version": target_definition_version(config),
	}


def _with_sequence_metadata(row: Mapping[str, Any], sample_index: int, config: Mapping[str, Any]) -> dict[str, Any]:
	item = dict(row)
	item.update(_sequence_metadata(sample_index, config))
	return item


def _complete_metadata_rows_from_npz_shard(
	shard_path: Path,
	split: str,
	existing_rows: Sequence[Mapping[str, Any]],
	config: Mapping[str, Any],
) -> list[dict[str, Any]]:
	with np.load(shard_path, allow_pickle=False) as shard:
		if "X" not in shard.files or "y" not in shard.files:
			raise ValueError(f"Shard is missing required X/y arrays: {shard_path}")
		x_shape = tuple(int(value) for value in shard["X"].shape)
		y_shape = tuple(int(value) for value in shard["y"].shape)
		num_samples = int(x_shape[0])
		sample_ids = [str(value) for value in shard["sample_ids"]] if "sample_ids" in shard.files else None
		dataset_ids = [int(value) for value in shard["dataset_ids"]] if "dataset_ids" in shard.files else None
		patch_y0_values = [int(value) for value in shard["patch_y0"]] if "patch_y0" in shard.files else None
		patch_x0_values = [int(value) for value in shard["patch_x0"]] if "patch_x0" in shard.files else None
		sample_indices = [int(value) for value in shard["sample_indices"]] if "sample_indices" in shard.files else None

	rows_by_local_index = {
		int(row.get("local_index", offset)): dict(row)
		for offset, row in enumerate(existing_rows)
	}
	shard_name = shard_path.name
	completed: list[dict[str, Any]] = []
	for local_index in range(num_samples):
		if local_index in rows_by_local_index:
			item = dict(rows_by_local_index[local_index])
			sample_index = int(item.get("sample_index", item.get("start_idx", -1)))
			if sample_index >= 0:
				item.update(_sequence_metadata(sample_index, config))
			item["shard"] = f"{split}/{shard_name}"
			item["local_index"] = int(local_index)
			completed.append(item)
			continue

		sample_id = sample_ids[local_index] if sample_ids is not None else ""
		fire_name, parsed_sample_index, parsed_y0, parsed_x0 = _parse_sample_id(sample_id)
		sample_index = sample_indices[local_index] if sample_indices is not None else parsed_sample_index
		patch_y0 = patch_y0_values[local_index] if patch_y0_values is not None else parsed_y0
		patch_x0 = patch_x0_values[local_index] if patch_x0_values is not None else parsed_x0
		if sample_index is None or patch_y0 is None or patch_x0 is None:
			raise ValueError(f"Cannot reconstruct metadata for {shard_path} local_index={local_index}.")
		patch_h = int(y_shape[-2])
		patch_w = int(y_shape[-1])
		item = {
			"split": split,
			"fire_name": fire_name,
			"dataset_id": int(dataset_ids[local_index]) if dataset_ids is not None else -1,
			"sample_index": int(sample_index),
			"patch": {
				"y0": int(patch_y0),
				"y1": int(patch_y0) + patch_h,
				"x0": int(patch_x0),
				"x1": int(patch_x0) + patch_w,
			},
			"patch_type": "reconstructed",
			"x_shape": [int(value) for value in x_shape[1:]],
			"y_shape": [int(value) for value in y_shape[1:]],
			"sample_id": sample_id or f"{split}:{fire_name}:{sample_index}:{patch_y0}:{patch_x0}",
			"shard": f"{split}/{shard_name}",
			"local_index": int(local_index),
		}
		completed.append(_with_sequence_metadata(item, int(sample_index), config))
	return completed


def _complete_metadata_rows_from_shard(
	shard_path: Path,
	shard_format: str,
	split: str,
	existing_rows: Sequence[Mapping[str, Any]],
	config: Mapping[str, Any],
) -> list[dict[str, Any]]:
	if str(shard_format).lower() != "npz":
		return _normalized_shard_metadata_rows(existing_rows, split, shard_path.name)
	return _complete_metadata_rows_from_npz_shard(shard_path, split, existing_rows, config)


def _materialize_legacy_shard_metadata(
	split_dir: Path,
	split: str,
	shard_format: str,
	config: Mapping[str, Any],
) -> None:
	"""Create per-shard metadata files from a legacy split-level metadata.jsonl."""

	aggregate_metadata_path = split_dir / "metadata.jsonl"
	if not aggregate_metadata_path.exists():
		return
	aggregate_rows = _read_metadata_rows(aggregate_metadata_path)
	rows_by_shard = _group_metadata_rows_by_shard(aggregate_rows)
	samples_per_shard = int(_get_section(config, "cache").get("samples_per_shard", 512))
	shard_index = 0
	while True:
		shard_path, metadata_path = _shard_artifacts(split_dir, shard_index, shard_format)
		if not shard_path.exists():
			break
		if metadata_path.exists():
			shard_index += 1
			continue
		shard_rows = rows_by_shard.get(shard_path.name, [])
		if not shard_rows:
			break
		if len(shard_rows) >= samples_per_shard:
			completed_rows = _normalized_shard_metadata_rows(
				shard_rows[:samples_per_shard],
				split,
				shard_path.name,
			)
		else:
			completed_rows = _complete_metadata_rows_from_shard(
				shard_path,
				shard_format,
				split,
				shard_rows,
				config,
			)
			if len(completed_rows) <= len(shard_rows):
				break
		_atomic_write_text(
			metadata_path,
			"".join(json.dumps(row, sort_keys=True) + "\n" for row in completed_rows),
		)
		shard_index += 1


def _discover_completed_shards(
	split_dir: Path,
	shard_format: str,
	*,
	validate_shard_shapes: bool = False,
	config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
	completed: list[dict[str, Any]] = []
	if not split_dir.exists():
		return completed
	shard_index = 0
	while True:
		shard_path, metadata_path = _shard_artifacts(split_dir, shard_index, shard_format)
		if not shard_path.exists() or not metadata_path.exists():
			break
		metadata_rows = _read_metadata_rows(metadata_path)
		if not metadata_rows:
			break
		if config is not None:
			enriched_rows: list[dict[str, Any]] = []
			changed = False
			for row in metadata_rows:
				item = dict(row)
				sample_index = item.get("sample_index", item.get("start_idx"))
				try:
					sample_index_int = int(sample_index)
				except (TypeError, ValueError):
					enriched_rows.append(item)
					continue
				enriched = _with_sequence_metadata(item, sample_index_int, config)
				changed = changed or enriched != item
				enriched_rows.append(enriched)
			if changed:
				_atomic_write_text(
					metadata_path,
					"".join(json.dumps(row, sort_keys=True) + "\n" for row in enriched_rows),
				)
			metadata_rows = enriched_rows
		num_samples = int(len(metadata_rows))
		if validate_shard_shapes:
			shard_samples = _read_shard_num_samples(shard_path, shard_format)
			if shard_samples != num_samples:
				raise ValueError(
					f"Shard metadata length mismatch for {shard_path}: shard samples={shard_samples}, metadata rows={num_samples}."
				)
		completed.append(
			{
				"index": shard_index,
				"path": shard_path,
				"metadata_path": metadata_path,
				"num_samples": num_samples,
				"metadata_rows": metadata_rows,
			}
		)
		shard_index += 1
	return completed


def _cleanup_incomplete_shard_artifacts(split_dir: Path, start_index: int, shard_format: str) -> None:
	if not split_dir.exists():
		return
	for path in split_dir.glob("shard_*"):
		if not path.is_file():
			continue
		if path.suffix.lower() not in {".npz", ".pt", ".jsonl"}:
			continue
		if not path.name.startswith("shard_"):
			continue
		if path.suffix.lower() == ".jsonl":
			prefix = path.name.split(".metadata", 1)[0]
			try:
				shard_index = int(prefix.split("_", 1)[1])
			except (IndexError, ValueError):
				continue
		else:
			if path.suffix.lower() != (".pt" if str(shard_format).lower() == "pt" else ".npz"):
				continue
			shard_index = _shard_index_from_path(path)
		if shard_index >= start_index:
			path.unlink(missing_ok=True)


def _rebuild_split_metadata_jsonl(split_dir: Path) -> int:
	rows: list[dict[str, Any]] = []
	for metadata_path in sorted(split_dir.glob("shard_*.metadata.jsonl")):
		rows.extend(_read_metadata_rows(metadata_path))
	output_path = split_dir / "metadata.jsonl"
	with output_path.open("w", encoding="utf-8") as handle:
		for row in rows:
			handle.write(json.dumps(row, sort_keys=True) + "\n")
	return len(rows)


def _checkpoint_path(split_dir: Path) -> Path:
	return split_dir / "resume_checkpoint.json"


def _load_resume_checkpoint(split_dir: Path) -> dict[str, Any] | None:
	path = _checkpoint_path(split_dir)
	if not path.exists():
		return None
	with path.open("r", encoding="utf-8") as handle:
		checkpoint = json.load(handle)
	if not isinstance(checkpoint, dict):
		raise ValueError(f"Resume checkpoint must contain a JSON object: {path}")
	return checkpoint


def _save_resume_checkpoint(split_dir: Path, split: str, next_episode: int, next_shard_index: int, total_samples: int) -> None:
	path = _checkpoint_path(split_dir)
	temporary_path = path.with_name(f".{path.name}.tmp")
	payload = {
		"split": split,
		"next_episode": int(next_episode),
		"next_shard_index": int(next_shard_index),
		"total_samples": int(total_samples),
		"updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
	}
	with temporary_path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, sort_keys=True)
		handle.write("\n")
	temporary_path.replace(path)


def _build_split_refs(
	config: Mapping[str, Any],
	dataset_records: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
	split_mode = str(config.get("split_mode", "train_val_test")).lower()
	if split_mode == "multi_dataset_chronological":
		split_mode = "multi_fire_chronological"
	if split_mode == "manual_fire_holdout":
		manual_section = _get_section(config, "manual_fire_split")
		sample_refs = manual_fire_holdout_splits(
			dataset_records=dataset_records,
			train_fire_names=manual_section.get("train_fires", []),
			val_fire_names=manual_section.get("val_fires", []),
			test_fire_names=manual_section.get("test_fires", []),
			input_sequence_length=int(config["input_sequence_length"]),
			prediction_horizon=int(config["prediction_horizon"]),
			config=config,
		)
	elif split_mode == "multi_fire_chronological":
		sample_refs = multi_fire_chronological_splits(
			dataset_records=dataset_records,
			input_sequence_length=int(config["input_sequence_length"]),
			prediction_horizon=int(config["prediction_horizon"]),
			train_fraction=float(config.get("train_fraction", 0.7)),
			val_fraction=float(config.get("val_fraction", 0.15)),
			test_fraction=float(config.get("test_fraction", 0.15)),
		)
	else:
		raise ValueError(
			"Patch-cache precompute currently supports split_mode=manual_fire_holdout "
			"or multi_fire_chronological."
		)

	patching = resolve_patching_config(config)
	if bool(patching["enabled"]):
		sample_refs = {
			split_name: build_sliding_patch_refs_for_split(
				dataset_records=dataset_records,
				sample_refs=sample_refs[split_name],
				split=split_name,
				config=config,
			)
			if resolve_split_patch_mode(config, split_name) == "sliding_window"
			else list(sample_refs[split_name])
			for split_name in ("train", "val", "test")
		}
	return sample_refs


def _build_dataset(
	config: Mapping[str, Any],
	dataset_records: Sequence[Mapping[str, Any]],
	sample_refs: Sequence[Mapping[str, Any]],
	split: str,
	normalization_stats: str | Path | None,
) -> MultiFirePatchSequenceDataset:
	patching = resolve_patching_config(config)
	target_normalization = _get_section(config, "target_normalization")
	return MultiFirePatchSequenceDataset(
		dataset_records=dataset_records,
		sample_refs=sample_refs,
		input_sequence_length=int(config["input_sequence_length"]),
		prediction_horizon=int(config["prediction_horizon"]),
		target_channel=int(config.get("target_channel", 0)),
		input_channel_count=int(config.get("input_channel_count", _get_section(config, "model").get("input_channels", 0))),
		input_channel_indices=config.get("input_channel_indices"),
		task_type=str(config.get("task_type", _get_section(config, "training").get("task_type", "regression"))),
		fire_threshold=float(config.get("fire_threshold", _get_section(config, "training").get("fire_threshold", 0.5))),
		use_patches=bool(patching["enabled"]),
		patch_size=int(patching["patch_height"]),
		active_patch_probability=float(patching["active_patch_probability"]),
		active_threshold=float(config.get("active_threshold", config.get("fire_threshold", 0.5))),
		normalization_stats=normalization_stats,
		normalize_target=bool(target_normalization.get("enabled", False)),
		return_metadata=True,
		config=config,
		split=split,
	)


def _base_manifest(
	config: Mapping[str, Any],
	dataset_records: Sequence[Mapping[str, Any]],
	split_refs: Mapping[str, Sequence[Mapping[str, Any]]],
	dataset: MultiFirePatchSequenceDataset,
) -> dict[str, Any]:
	patching = resolve_patching_config(config)
	energy_release = resolve_energy_release_config(config)
	energy_names = resolve_energy_output_channel_names(config)
	cache_config = _get_section(config, "cache")
	target_offsets = temporal_target_offsets(config)
	dataset_index_path = resolve_dataset_index_path(config)
	trim_manifest = extract_temporal_trim_manifest(dataset_records)
	config_provenance = _config_provenance(config)
	return {
		"cache_version": str(cache_config.get("cache_version", "v1")),
		"created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
		"config_hash": compute_cache_config_hash(config),
		"config_provenance": config_provenance,
		"config_path": config_provenance["config_path"],
		"config_sha256": config_provenance["config_sha256"],
		"resolved_config_sha256": config_provenance["resolved_config_sha256"],
		"base_config_path": config_provenance["base_config_path"],
		"base_config_sha256": config_provenance["base_config_sha256"],
		"experiment_name": config_provenance["experiment_name"],
		"dataset_index_path": str(dataset_index_path) if dataset_index_path is not None else None,
		"trimmed_index_path": str(dataset_index_path) if dataset_index_path is not None else None,
		"dataset_index_hash": compute_dataset_index_hash(config),
		"trim_metadata_hash": compute_current_trim_metadata_hash(config),
		"temporal_trim_enabled": bool(trim_manifest["temporal_trim_enabled"]),
		"fires": trim_manifest["fires"],
		"split_mode": str(config.get("split_mode", "train_val_test")),
		"manual_fire_split": _get_section(config, "manual_fire_split"),
		"input_sequence_length": int(config["input_sequence_length"]),
		"prediction_horizon": int(config["prediction_horizon"]),
		"target_offset_from_start": int(target_offsets["target_offset_from_start"]),
		"target_offset_from_last_input": int(target_offsets["target_offset_from_last_input"]),
		"target_definition_version": target_definition_version(config),
		"input_channels": int(dataset.total_input_channels),
		"base_input_channel_count": int(dataset.base_input_channel_count),
		"fuel_flux_engineered_channel_count": int(dataset.fuel_flux_engineered_channel_count),
		"atmospheric_engineered_channel_count": int(dataset.atmospheric_engineered_channel_count),
		"energy_history_channel_count": int(dataset.energy_history_channel_count),
		"engineered_channel_count": int(dataset.engineered_channel_count),
		"output_channels": int(_get_section(config, "model").get("output_channels", 1)),
		"patch_height": int(patching["patch_height"]),
		"patch_width": int(patching["patch_width"]),
		"include_border_patches": bool(patching["include_border_patches"]),
		"patch_modes": {
			"train": resolve_split_patch_mode(config, "train"),
			"val": resolve_split_patch_mode(config, "val"),
			"test": resolve_split_patch_mode(config, "test"),
		},
		"strides": {
			"train": resolve_split_patch_stride(config, "train"),
			"val": resolve_split_patch_stride(config, "val"),
			"test": resolve_split_patch_stride(config, "test"),
		},
		"energy_release_enabled": bool(energy_release["enabled"]),
		"energy_release_channel": 3 if energy_names else None,
		"energy_output_channel_names": list(energy_names),
		"target_transform": str(energy_release["target_transform"]),
		"target_normalization_enabled": bool(_get_section(config, "target_normalization").get("enabled", False)),
		"save_normalized_inputs": bool(cache_config.get("save_normalized_inputs", False)),
		"shard_format": str(cache_config.get("shard_format", "npz")).lower(),
		"compressed": bool(cache_config.get("compressed", False)),
		"samples_per_shard": int(cache_config.get("samples_per_shard", 512)),
		"num_train_patches": 0,
		"num_val_patches": 0,
		"num_test_patches": 0,
		"shards": {"train": [], "val": [], "test": []},
		"fires_by_split": {
			split: sorted({str(ref.get("dataset_name", dataset_records[int(ref["dataset_id"])]["dataset_name"])) for ref in refs})
			for split, refs in split_refs.items()
		},
		"dataset_records": [
			{
				"dataset_id": int(record["dataset_id"]),
				"dataset_name": str(record["dataset_name"]),
				"data_dir": str(record["data_dir"]),
				"num_files": int(record["num_files"]),
				"effective_num_files": int(record.get("effective_num_files", record["num_files"])),
				"temporal_trim": dict(record.get("temporal_trim", {})),
				"raw_shape": [int(value) for value in record["raw_shape"]],
			}
			for record in dataset_records
		],
	}


def _patch_from_metadata(metadata: Mapping[str, Any], y_array: np.ndarray) -> dict[str, int]:
	patch = metadata.get("patch")
	if isinstance(patch, Mapping):
		return {key: int(patch[key]) for key in ("y0", "y1", "x0", "x1")}
	patch_top = int(metadata.get("patch_top", 0))
	patch_left = int(metadata.get("patch_left", 0))
	height = int(y_array.shape[-2])
	width = int(y_array.shape[-1])
	return {"y0": patch_top, "y1": patch_top + height, "x0": patch_left, "x1": patch_left + width}


def _patch_type(split: str, metadata: Mapping[str, Any], y_array: np.ndarray, config: Mapping[str, Any]) -> str:
	if isinstance(metadata.get("patch"), Mapping) and split in {"val", "test"}:
		return "sliding"
	if y_array.ndim == 3 and y_array.shape[0] >= 3:
		consumed_threshold = float(resolve_patching_config(config)["consumed_active_threshold"])
		mask_active = float(np.mean(np.asarray(y_array[2]) > 0.5))
		consumed_max = float(max(np.max(y_array[0]), np.max(y_array[1])))
		if mask_active > 0.0 or consumed_max > consumed_threshold:
			return "active"
	return "random"


def _metadata_row(
	split: str,
	x_array: np.ndarray,
	y_array: np.ndarray,
	metadata: Mapping[str, Any],
	config: Mapping[str, Any],
) -> dict[str, Any]:
	patch = _patch_from_metadata(metadata, y_array)
	sequence_row = _sequence_metadata(int(metadata.get("sample_index", -1)), config)
	for key in (
		"start_idx",
		"input_indices",
		"last_input_idx",
		"target_idx",
		"current_idx",
		"future_idx",
		"current_index",
		"future_index",
		"local_start_idx",
		"local_input_indices",
		"local_last_input_idx",
		"local_target_idx",
		"original_start_idx",
		"original_input_indices",
		"original_last_input_idx",
		"original_target_idx",
		"trim_start_index",
		"trim_end_index",
		"trimmed_num_frames",
		"original_num_frames",
		"temporal_trim_enabled",
	):
		if key in metadata:
			sequence_row[key] = metadata[key]
	row = {
		"split": split,
		"fire_name": str(metadata.get("dataset_name", metadata.get("data_dir", ""))),
		"dataset_id": int(metadata.get("dataset_id", -1)),
		"sample_index": int(metadata.get("sample_index", -1)),
		**sequence_row,
		"patch": patch,
		"patch_type": _patch_type(split, metadata, y_array, config),
		"x_shape": [int(value) for value in x_array.shape],
		"y_shape": [int(value) for value in y_array.shape],
		"energy_release_log_min": float(np.min(y_array[3])) if y_array.ndim == 3 and y_array.shape[0] > 3 else None,
		"energy_release_log_max": float(np.max(y_array[3])) if y_array.ndim == 3 and y_array.shape[0] > 3 else None,
		"mask_active_fraction": float(np.mean(y_array[2] > 0.5)) if y_array.ndim == 3 and y_array.shape[0] > 2 else None,
		"surface_consumed_max": float(np.max(y_array[0])) if y_array.ndim == 3 and y_array.shape[0] > 0 else None,
		"canopy_consumed_max": float(np.max(y_array[1])) if y_array.ndim == 3 and y_array.shape[0] > 1 else None,
	}
	row["sample_id"] = (
		f"{split}:{row['fire_name']}:{row['sample_index']}:"
		f"{patch['y0']}:{patch['x0']}"
	)
	return row


def _base_channel_position(config: Mapping[str, Any], raw_channel: int) -> int | None:
	input_channel_count = int(config.get("input_channel_count", _get_section(config, "model").get("input_channels", 0)))
	configured = config.get("input_channel_indices")
	if configured is None:
		return int(raw_channel) if 0 <= int(raw_channel) < input_channel_count else None
	indices = [int(value) for value in configured]
	try:
		return indices.index(int(raw_channel))
	except ValueError:
		return None


def _save_preview(
	x_array: np.ndarray,
	y_array: np.ndarray,
	metadata: Mapping[str, Any],
	config: Mapping[str, Any],
	output_path: Path,
) -> None:
	plt = _get_pyplot()
	if plt is None:
		return
	layout = _get_section(config, "channel_layout")
	surface_pos = _base_channel_position(config, int(layout.get("surface_fuel_channel", 84)))
	canopy_pos = _base_channel_position(config, int(layout.get("canopy_fuel_channel", 85)))
	latest_x = x_array[-1]
	panels: list[tuple[str, np.ndarray]] = []
	if surface_pos is not None and surface_pos < latest_x.shape[0]:
		panels.append(("latest surface fuel", latest_x[surface_pos]))
	if canopy_pos is not None and canopy_pos < latest_x.shape[0]:
		panels.append(("latest canopy fuel", latest_x[canopy_pos]))
	if y_array.shape[0] > 0:
		panels.append(("target surface consumed", y_array[0]))
	if y_array.shape[0] > 1:
		panels.append(("target canopy consumed", y_array[1]))
	if y_array.shape[0] > 2:
		panels.append(("target mask", y_array[2]))
	if y_array.shape[0] > 3:
		panels.append(("target log1p energy", y_array[3]))
	if not panels:
		return

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig, axes = plt.subplots(2, 3, figsize=(10, 6), dpi=140, constrained_layout=True)
	flat_axes = list(axes.ravel())
	for axis, (title, array) in zip(flat_axes, panels):
		image = axis.imshow(array, cmap="viridis")
		axis.set_title(title, fontsize=8)
		axis.set_xticks([])
		axis.set_yticks([])
		fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
	for axis in flat_axes[len(panels):]:
		axis.axis("off")
	patch = metadata.get("patch", {})
	fig.suptitle(
		f"{metadata.get('fire_name', metadata.get('dataset_name', 'fire'))} "
		f"sample={metadata.get('sample_index')} patch={patch}",
		fontsize=9,
	)
	fig.savefig(output_path, bbox_inches="tight")
	plt.close(fig)


def _write_shard(
	split_dir: Path,
	split: str,
	shard_index: int,
	x_items: list[np.ndarray],
	y_items: list[np.ndarray],
	metadata_items: list[dict[str, Any]],
	cache_config: Mapping[str, Any],
) -> dict[str, Any]:
	shard_format = str(cache_config.get("shard_format", "npz")).lower()
	compressed = bool(cache_config.get("compressed", False))
	shard_path, metadata_path = _shard_artifacts(split_dir, shard_index, shard_format)
	temporary_shard_path = shard_path.with_name(f".{shard_path.name}.tmp")
	temporary_metadata_path = metadata_path.with_name(f".{metadata_path.name}.tmp")
	x_array = np.stack(x_items, axis=0).astype(np.float32, copy=False)
	y_array = np.stack(y_items, axis=0).astype(np.float32, copy=False)
	sample_ids = np.asarray([str(item["sample_id"]) for item in metadata_items])
	dataset_ids = np.asarray([int(item["dataset_id"]) for item in metadata_items], dtype=np.int64)
	patch_y0 = np.asarray([int(item["patch"]["y0"]) for item in metadata_items], dtype=np.int64)
	patch_x0 = np.asarray([int(item["patch"]["x0"]) for item in metadata_items], dtype=np.int64)
	sample_indices = np.asarray([int(item["sample_index"]) for item in metadata_items], dtype=np.int64)

	if shard_format == "npz":
		save_fn = np.savez_compressed if compressed else np.savez
		with temporary_shard_path.open("wb") as handle:
			save_fn(
				handle,
				X=x_array,
				y=y_array,
				sample_ids=sample_ids,
				dataset_ids=dataset_ids,
				patch_y0=patch_y0,
				patch_x0=patch_x0,
				sample_indices=sample_indices,
			)
	elif shard_format == "pt":
		if torch is None:
			raise ImportError("PyTorch is required to write cache.shard_format=pt shards.")
		with temporary_shard_path.open("wb") as handle:
			torch.save(
				{
					"X": torch.from_numpy(x_array),
					"y": torch.from_numpy(y_array),
					"sample_ids": sample_ids.tolist(),
					"dataset_ids": torch.from_numpy(dataset_ids),
					"patch_y0": torch.from_numpy(patch_y0),
					"patch_x0": torch.from_numpy(patch_x0),
					"sample_indices": torch.from_numpy(sample_indices),
				},
				handle,
			)
	else:
		raise ValueError(f"Unsupported cache.shard_format={shard_format!r}. Expected 'npz' or 'pt'.")

	metadata_lines: list[str] = []
	shard_name = shard_path.name
	for local_index, item in enumerate(metadata_items):
		item = dict(item)
		item["shard"] = f"{split}/{shard_name}"
		item["local_index"] = int(local_index)
		metadata_lines.append(json.dumps(item, sort_keys=True))
	_atomic_write_text(temporary_metadata_path, "\n".join(metadata_lines) + "\n")
	temporary_shard_path.replace(shard_path)
	temporary_metadata_path.replace(metadata_path)

	return {
		"path": f"{split}/{shard_name}",
		"num_samples": int(x_array.shape[0]),
		"x_shape": [int(value) for value in x_array.shape],
		"y_shape": [int(value) for value in y_array.shape],
		"bytes": int(x_array.nbytes + y_array.nbytes),
	}


def _rebuild_summary_csv(cache_dir: Path) -> None:
	output_path = cache_dir / "patch_summary.csv"
	fieldnames = [
		"split",
		"fire_name",
		"shard",
		"sample_index",
		"start_idx",
		"local_start_idx",
		"original_start_idx",
		"last_input_idx",
		"target_idx",
		"original_last_input_idx",
		"original_target_idx",
		"trim_start_index",
		"trim_end_index",
		"current_idx",
		"future_idx",
		"input_sequence_length",
		"prediction_horizon",
		"patch_y0",
		"patch_x0",
		"patch_type",
		"mask_active_fraction",
		"max_surface_consumed",
		"max_canopy_consumed",
	]
	rows: list[dict[str, Any]] = []
	for split in ("train", "val", "test"):
		metadata_path = cache_dir / split / "metadata.jsonl"
		if not metadata_path.exists():
			continue
		with metadata_path.open("r", encoding="utf-8") as handle:
			for line in handle:
				if not line.strip():
					continue
				item = json.loads(line)
				patch = item.get("patch", {})
				rows.append(
					{
						"split": split,
						"fire_name": item.get("fire_name"),
						"shard": item.get("shard"),
						"sample_index": item.get("sample_index"),
						"start_idx": item.get("start_idx"),
						"local_start_idx": item.get("local_start_idx"),
						"original_start_idx": item.get("original_start_idx"),
						"last_input_idx": item.get("last_input_idx"),
						"target_idx": item.get("target_idx"),
						"original_last_input_idx": item.get("original_last_input_idx"),
						"original_target_idx": item.get("original_target_idx"),
						"trim_start_index": item.get("trim_start_index"),
						"trim_end_index": item.get("trim_end_index"),
						"current_idx": item.get("current_idx"),
						"future_idx": item.get("future_idx"),
						"input_sequence_length": item.get("input_sequence_length"),
						"prediction_horizon": item.get("prediction_horizon"),
						"patch_y0": patch.get("y0"),
						"patch_x0": patch.get("x0"),
						"patch_type": item.get("patch_type"),
						"mask_active_fraction": item.get("mask_active_fraction"),
						"max_surface_consumed": item.get("surface_consumed_max"),
						"max_canopy_consumed": item.get("canopy_consumed_max"),
					}
				)
	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _precompute_split(
	config: Mapping[str, Any],
	dataset: MultiFirePatchSequenceDataset,
	split: str,
	cache_dir: Path,
	manifest: dict[str, Any],
	resume_from_episode: int | None = None,
) -> None:
	cache_config = _get_section(config, "cache")
	samples_per_shard = int(cache_config.get("samples_per_shard", 512))
	if samples_per_shard <= 0:
		raise ValueError(f"cache.samples_per_shard must be positive, got {samples_per_shard}.")
	split_dir = cache_dir / split
	split_dir.mkdir(parents=True, exist_ok=True)
	preview_enabled = bool(cache_config.get("save_preview_images", True))
	max_previews = int(cache_config.get("num_preview_images", 50))
	save_metadata = bool(cache_config.get("save_metadata", True))
	preview_dir = cache_dir / "previews" / split
	shard_format = str(cache_config.get("shard_format", "npz")).lower()
	if save_metadata:
		_materialize_legacy_shard_metadata(split_dir, split, shard_format, config)
	existing_shards = _discover_completed_shards(split_dir, shard_format, config=config)
	existing_manifest_shards = [
		{
			"path": f"{split}/{Path(str(item['path'])).name}",
			"num_samples": int(item["num_samples"]),
		}
		for item in existing_shards
	]
	resume_index = len(existing_shards)
	resume_sample_index = int(sum(int(item["num_samples"]) for item in existing_shards))
	if resume_sample_index > len(dataset):
		raise RuntimeError(
			f"{split}: found {resume_sample_index} cached patch(es), but the current dataset only has "
			f"{len(dataset)} patch(es). Pass --overwrite if the split definition changed."
		)
	checkpoint = _load_resume_checkpoint(split_dir)
	checkpoint_episode = None
	if checkpoint is not None:
		checkpoint_split = str(checkpoint.get("split", split))
		if checkpoint_split != split:
			print(f"{split}: ignoring resume checkpoint for split={checkpoint_split!r}.")
			checkpoint = None
		else:
			checkpoint_shard_index = int(checkpoint.get("next_shard_index", resume_index))
			if checkpoint_shard_index != resume_index:
				print(
					f"{split}: checkpoint next_shard_index={checkpoint_shard_index} differs from "
					f"complete shard count={resume_index}; using complete shards as the resume point."
				)
	if checkpoint is not None:
		checkpoint_episode = int(checkpoint.get("next_episode", 0))
	if resume_from_episode is not None:
		if resume_from_episode < 0:
			raise ValueError(f"resume_from_episode must be non-negative, got {resume_from_episode}.")
		if int(resume_from_episode) > resume_sample_index:
			raise RuntimeError(
				f"{split}: --resume-from-episode={resume_from_episode} is ahead of the last complete "
				f"cached patch index {resume_sample_index}. Reprocessing from {resume_sample_index} "
				"keeps the cache contiguous; lower the requested index or pass --overwrite to rebuild."
			)
		start_episode = resume_sample_index
		if int(resume_from_episode) < resume_sample_index:
			print(
				f"{split}: requested resume episode/sample index {resume_from_episode}; "
				f"complete shards already cover {resume_sample_index} patch(es)."
			)
		else:
			print(f"{split}: starting from requested episode/sample index {start_episode}.")
	elif checkpoint_episode is not None:
		start_episode = resume_sample_index
		if checkpoint_episode != resume_sample_index:
			print(
				f"{split}: checkpoint next_episode={checkpoint_episode} differs from "
				f"{resume_sample_index} complete cached patch(es); using complete shards as the resume point."
			)
		else:
			print(f"{split}: resuming from checkpoint episode/sample index {start_episode}.")
	else:
		start_episode = resume_sample_index
	if resume_index > 0:
		_cleanup_incomplete_shard_artifacts(split_dir, resume_index, shard_format)
		print(f"{split}: resuming from shard {resume_index:06d} after {resume_sample_index} cached patch(es).")
	if start_episode > resume_sample_index:
		print(f"{split}: skipping ahead to episode/sample index {start_episode}.")

	shards: list[dict[str, Any]] = []
	x_buffer: list[np.ndarray] = []
	y_buffer: list[np.ndarray] = []
	metadata_buffer: list[dict[str, Any]] = []
	preview_count = len(list(preview_dir.glob("preview_*.png"))) if preview_enabled else 0
	shard_index = resume_index
	total_samples = start_episode
	warned_large_shard = False
	if total_samples < len(dataset):
		for item_index in range(total_samples, len(dataset)):
			x_tensor, y_tensor, metadata = dataset[item_index]
			x_array = x_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
			y_array = y_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
			if not warned_large_shard:
				estimated_shard_gb = (x_array.nbytes + y_array.nbytes) * samples_per_shard / (1024.0 ** 3)
				if estimated_shard_gb > 2.0:
					print(
						f"WARNING: cache.samples_per_shard={samples_per_shard} will buffer about "
						f"{estimated_shard_gb:.2f} GiB per shard before writing. "
						"Lower cache.samples_per_shard if the precompute job runs out of memory."
					)
				warned_large_shard = True
			row = _metadata_row(split, x_array, y_array, metadata, config)
			x_buffer.append(x_array)
			y_buffer.append(y_array)
			metadata_buffer.append(row)
			total_samples += 1

			if preview_enabled and preview_count < max_previews:
				preview_path = cache_dir / "previews" / split / f"preview_{preview_count:04d}.png"
				_save_preview(x_array, y_array, row, config, preview_path)
				preview_count += 1

			if len(x_buffer) >= samples_per_shard:
				shards.append(_write_shard(split_dir, split, shard_index, x_buffer, y_buffer, metadata_buffer, cache_config))
				shard_index += 1
				x_buffer.clear()
				y_buffer.clear()
				metadata_buffer.clear()
				_save_resume_checkpoint(split_dir, split, total_samples, shard_index, total_samples)
			if total_samples % 100 == 0:
				print(f"{split}: cached {total_samples}/{len(dataset)} patches (next episode {total_samples})")

		if x_buffer:
			shards.append(_write_shard(split_dir, split, shard_index, x_buffer, y_buffer, metadata_buffer, cache_config))
			shard_index += 1
			x_buffer.clear()
			y_buffer.clear()
			metadata_buffer.clear()
			_save_resume_checkpoint(split_dir, split, total_samples, shard_index, total_samples)

	shards = existing_manifest_shards + shards
	shard_sample_count = int(sum(int(item["num_samples"]) for item in shards))
	if shard_sample_count != total_samples:
		raise RuntimeError(
			f"{split}: internal resume mismatch: manifest would report {total_samples} patch(es), "
			f"but complete shards contain {shard_sample_count} patch(es)."
		)
	if save_metadata:
		metadata_rows = _rebuild_split_metadata_jsonl(split_dir)
		if metadata_rows != total_samples:
			raise RuntimeError(
				f"{split}: metadata row count mismatch after resume: rows={metadata_rows}, "
				f"cached patches={total_samples}."
			)
	else:
		(split_dir / "metadata.jsonl").unlink(missing_ok=True)
	_save_resume_checkpoint(split_dir, split, total_samples, shard_index, total_samples)

	manifest["shards"][split] = shards
	manifest[f"num_{split}_patches"] = int(total_samples)
	print(f"{split}: wrote {total_samples} patches across {len(shards)} shard(s) to {split_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Precompute wildfire ConvLSTM patch-cache shards.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"], help="Split to precompute.")
	parser.add_argument(
		"--resume-from-episode",
		type=int,
		default=None,
		help="Resume at or after this episode/sample index within the selected split.",
	)
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing shard files for the selected split(s).")
	parser.add_argument("--force", action="store_true", help="Overwrite existing shards and take ownership of an existing cache lock.")
	parser.add_argument("--ignore_stale_lock", action="store_true", help="Replace an abandoned cache lock before precomputing.")
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	config_path = Path(args.config).expanduser().resolve()
	config = _ensure_config_path(load_config(config_path), config_path)
	cache_config = _get_section(config, "cache")
	if bool(_get_section(config, "target_normalization").get("enabled", False)):
		raise ValueError("Patch cache currently expects target_normalization.enabled=false so cached y matches training targets.")

	cache_dir = get_patch_cache_dir(config)
	selected = _selected_splits(args.split)
	overwrite = bool(args.overwrite or args.force or cache_config.get("overwrite_existing", False))
	cache_dir.mkdir(parents=True, exist_ok=True)
	lock_path = _acquire_cache_lock(
		cache_dir,
		config,
		force=bool(args.force),
		ignore_stale_lock=bool(args.ignore_stale_lock),
	)
	atexit.register(_release_cache_lock, lock_path)

	for split in selected:
		split_dir = cache_dir / split
		if overwrite and split_dir.exists():
			shutil.rmtree(split_dir)

	save_normalized_inputs = bool(cache_config.get("save_normalized_inputs", False))
	normalization_stats = None
	if save_normalized_inputs:
		normalization_path = _get_section(config, "normalization").get("path")
		if normalization_path in (None, "", "null"):
			raise ValueError("cache.save_normalized_inputs=true requires normalization.path.")
		path = Path(normalization_path).expanduser()
		if not path.is_absolute():
			path = config_path.parent / path
		if not path.exists():
			raise FileNotFoundError(f"Normalization stats not found for normalized cache creation: {path}")
		normalization_stats = path

	dataset_records = discover_multiple_datasets(config)
	split_refs = _build_split_refs(config, dataset_records)
	first_dataset = _build_dataset(
		config=config,
		dataset_records=dataset_records,
		sample_refs=split_refs[selected[0]],
		split=selected[0],
		normalization_stats=normalization_stats,
	)
	manifest_path = cache_dir / MANIFEST_FILENAME
	rebuilding_all_splits = set(selected) == {"train", "val", "test"}
	base_manifest = _base_manifest(config, dataset_records, split_refs, first_dataset)
	if manifest_path.exists() and (not overwrite or not rebuilding_all_splits):
		manifest = load_cache_manifest(cache_dir)
		current_hash = compute_cache_config_hash(config)
		if str(manifest.get("config_hash")) != current_hash and not overwrite:
			raise RuntimeError(
				f"Existing manifest config hash differs from current config under {cache_dir}. "
				"Pass --overwrite if you intend to rebuild the cache."
			)
		for key in (
			"cache_version",
			"config_provenance",
			"config_path",
			"config_sha256",
			"resolved_config_sha256",
			"base_config_path",
			"base_config_sha256",
			"experiment_name",
			"dataset_index_path",
			"trimmed_index_path",
			"dataset_index_hash",
			"trim_metadata_hash",
			"temporal_trim_enabled",
			"fires",
			"split_mode",
			"manual_fire_split",
			"input_sequence_length",
			"prediction_horizon",
			"target_offset_from_start",
			"target_offset_from_last_input",
			"target_definition_version",
			"input_channels",
			"base_input_channel_count",
			"fuel_flux_engineered_channel_count",
			"atmospheric_engineered_channel_count",
			"energy_history_channel_count",
			"engineered_channel_count",
			"output_channels",
			"patch_height",
			"patch_width",
			"include_border_patches",
			"patch_modes",
			"strides",
			"energy_release_enabled",
			"energy_release_channel",
			"energy_output_channel_names",
			"target_transform",
			"target_normalization_enabled",
			"save_normalized_inputs",
			"shard_format",
			"compressed",
			"samples_per_shard",
			"fires_by_split",
			"dataset_records",
		):
			manifest[key] = base_manifest[key]
		manifest.setdefault("shards", {"train": [], "val": [], "test": []})
	else:
		manifest = base_manifest

	for split in selected:
		dataset = first_dataset if split == selected[0] else _build_dataset(
			config=config,
			dataset_records=dataset_records,
			sample_refs=split_refs[split],
			split=split,
			normalization_stats=normalization_stats,
		)
		_precompute_split(
			config,
			dataset,
			split,
			cache_dir,
			manifest,
			resume_from_episode=args.resume_from_episode,
		)
	print("Cache patch mode:")
	for split in ("train", "val", "test"):
		print(f"  {split}: {resolve_split_patch_mode(config, split)} stride={resolve_split_patch_stride(config, split)}")
	target_offsets = temporal_target_offsets(config)
	print(
		"Temporal target: "
		f"input_sequence_length={int(config['input_sequence_length'])} "
		f"prediction_horizon={int(config['prediction_horizon'])} "
		f"target_offset_from_start={target_offsets['target_offset_from_start']} "
		f"target_definition_version={target_definition_version(config)}"
	)

	manifest["created_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
	manifest["config_hash"] = compute_cache_config_hash(config)
	config_provenance = _config_provenance(config)
	manifest["config_provenance"] = config_provenance
	manifest["config_path"] = config_provenance["config_path"]
	manifest["config_sha256"] = config_provenance["config_sha256"]
	manifest["resolved_config_sha256"] = config_provenance["resolved_config_sha256"]
	manifest["base_config_path"] = config_provenance["base_config_path"]
	manifest["base_config_sha256"] = config_provenance["base_config_sha256"]
	manifest["experiment_name"] = config_provenance["experiment_name"]
	with manifest_path.open("w", encoding="utf-8") as handle:
		json.dump(manifest, handle, indent=2, sort_keys=True)
		handle.write("\n")
	_rebuild_summary_csv(cache_dir)
	_release_cache_lock(lock_path)
	print(f"Patch cache manifest: {manifest_path}")
	print(f"Patch summary CSV: {cache_dir / 'patch_summary.csv'}")


if __name__ == "__main__":
	main()
