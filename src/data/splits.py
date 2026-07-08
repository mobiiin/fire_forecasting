"""Dataset split helpers for chronological wildfire forecasting splits."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from src.data.patching import build_sliding_window_patches, resolve_patching_config


def _validate_nonnegative_fraction(name: str, fraction: float) -> None:
	"""Validate that one split fraction is non-negative."""

	if fraction < 0.0:
		raise ValueError(f"{name} must be non-negative, got {fraction}.")


def _validate_fractions(
	train_fraction: float,
	val_fraction: float,
	test_fraction: float,
) -> None:
	"""Validate split fractions before computing indices."""

	for name, fraction in (
		("train_fraction", train_fraction),
		("val_fraction", val_fraction),
		("test_fraction", test_fraction),
	):
		_validate_nonnegative_fraction(name, fraction)

	total = train_fraction + val_fraction + test_fraction
	if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
		raise ValueError(
			"Split fractions must sum to 1.0. "
			f"Got train={train_fraction}, val={val_fraction}, test={test_fraction}, total={total}."
		)


def _validate_train_val_fractions(
	train_fraction: float,
	val_fraction: float,
) -> None:
	"""Validate fractions for train/validation-only chronological splitting."""

	_validate_nonnegative_fraction("train_fraction", train_fraction)
	_validate_nonnegative_fraction("val_fraction", val_fraction)
	total = train_fraction + val_fraction
	if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
		raise ValueError(
			"For split_mode='train_val_external_test', train_fraction + val_fraction must sum to 1.0. "
			f"Got train={train_fraction}, val={val_fraction}, total={total}."
		)


def _sample_starts_for_segment(
	segment_start: int,
	segment_end: int,
	input_sequence_length: int,
	prediction_horizon: int,
) -> List[int]:
	"""Return valid sample start indices for a half-open raw-time segment."""

	latest_start = segment_end - input_sequence_length - prediction_horizon
	if latest_start < segment_start:
		return []
	return list(range(segment_start, latest_start + 1))


def chronological_train_val_split_indices(
	num_timesteps: int,
	input_sequence_length: int,
	prediction_horizon: int,
	train_fraction: float,
	val_fraction: float,
) -> Dict[str, List[int]]:
	"""Split forecasting sample indices chronologically into train and validation only."""

	if num_timesteps <= 0:
		raise ValueError(f"num_timesteps must be positive, got {num_timesteps}.")
	if input_sequence_length <= 0:
		raise ValueError(f"input_sequence_length must be positive, got {input_sequence_length}.")
	if prediction_horizon < 0:
		raise ValueError(f"prediction_horizon must be non-negative, got {prediction_horizon}.")

	_validate_train_val_fractions(train_fraction, val_fraction)

	max_valid_start = num_timesteps - input_sequence_length - prediction_horizon
	if max_valid_start < 0:
		raise ValueError(
			"Not enough timesteps to form a single valid sample. "
			f"Need at least input_sequence_length + prediction_horizon = "
			f"{input_sequence_length + prediction_horizon}, got {num_timesteps}."
		)

	train_length = int(math.floor(num_timesteps * train_fraction))
	val_segment_start = train_length
	train = _sample_starts_for_segment(
		segment_start=0,
		segment_end=val_segment_start,
		input_sequence_length=input_sequence_length,
		prediction_horizon=prediction_horizon,
	)
	val = _sample_starts_for_segment(
		segment_start=val_segment_start,
		segment_end=num_timesteps,
		input_sequence_length=input_sequence_length,
		prediction_horizon=prediction_horizon,
	)
	return {"train": train, "val": val}


def chronological_split_indices_for_dataset(
	num_timesteps: int,
	input_sequence_length: int,
	prediction_horizon: int,
	train_fraction: float,
	val_fraction: float,
	test_fraction: float,
) -> Dict[str, List[int]]:
	"""Split one dataset chronologically into train/val/test sample starts."""

	return chronological_split_indices(
		num_timesteps=num_timesteps,
		input_sequence_length=input_sequence_length,
		prediction_horizon=prediction_horizon,
		train_fraction=train_fraction,
		val_fraction=val_fraction,
		test_fraction=test_fraction,
		split_mode="train_val_test",
	)


def multi_dataset_chronological_splits(
	dataset_records: Sequence[Mapping[str, Any]],
	input_sequence_length: int,
	prediction_horizon: int,
	train_fraction: float,
	val_fraction: float,
	test_fraction: float,
) -> dict[str, list[dict[str, int]]]:
	"""Split each dataset independently, then concatenate split-local sample references."""

	_validate_fractions(train_fraction, val_fraction, test_fraction)
	combined: dict[str, list[dict[str, int]]] = {"train": [], "val": [], "test": []}
	rows: list[tuple[str, int, int, int, int]] = []

	for dataset_record in dataset_records:
		file_count = int(dataset_record["num_files"])
		dataset_id = int(dataset_record["dataset_id"])
		dataset_name = str(dataset_record["dataset_name"])
		splits = chronological_split_indices_for_dataset(
			num_timesteps=file_count,
			input_sequence_length=input_sequence_length,
			prediction_horizon=prediction_horizon,
			train_fraction=train_fraction,
			val_fraction=val_fraction,
			test_fraction=test_fraction,
		)
		rows.append((dataset_name, file_count, len(splits["train"]), len(splits["val"]), len(splits["test"])))
		for split_name in ("train", "val", "test"):
			for sample_index in splits[split_name]:
				combined[split_name].append({"dataset_id": dataset_id, "sample_index": int(sample_index)})

	print("dataset_name       files    train_samples    val_samples    test_samples")
	for dataset_name, file_count, train_count, val_count, test_count in rows:
		print(f"{dataset_name:<18} {file_count:<8} {train_count:<16} {val_count:<13} {test_count:<12}")
	print(
		f"{'TOTAL':<18} {'':<8} {len(combined['train']):<16} "
		f"{len(combined['val']):<13} {len(combined['test']):<12}"
	)
	return combined


def multi_fire_chronological_splits(
	dataset_records: Sequence[Mapping[str, Any]],
	input_sequence_length: int,
	prediction_horizon: int,
	train_fraction: float,
	val_fraction: float,
	test_fraction: float,
) -> dict[str, list[dict[str, int | str]]]:
	"""Split each fire independently, then concatenate split-local sample references."""

	base = multi_dataset_chronological_splits(
		dataset_records=dataset_records,
		input_sequence_length=input_sequence_length,
		prediction_horizon=prediction_horizon,
		train_fraction=train_fraction,
		val_fraction=val_fraction,
		test_fraction=test_fraction,
	)
	dataset_name_by_id = {
		int(record["dataset_id"]): str(record["dataset_name"])
		for record in dataset_records
	}
	annotated: dict[str, list[dict[str, int | str]]] = {"train": [], "val": [], "test": []}
	for split_name, refs in base.items():
		for ref in refs:
			dataset_id = int(ref["dataset_id"])
			annotated[split_name].append(
				{
					"dataset_id": dataset_id,
					"dataset_name": dataset_name_by_id[dataset_id],
					"sample_index": int(ref["sample_index"]),
					"fire_split_group": split_name,
				}
			)
	return annotated


def _take_fraction_of_indices(indices: Sequence[int], fraction: float, keep_from_end: bool = False) -> list[int]:
	"""Take a chronological subset of indices using a simple fraction."""

	if not 0.0 <= float(fraction) <= 1.0:
		raise ValueError(f"Split fraction must be in [0, 1], got {fraction}.")
	if not indices:
		return []
	if math.isclose(float(fraction), 1.0, rel_tol=1e-9, abs_tol=1e-9):
		return [int(index) for index in indices]
	count = max(1, int(math.floor(len(indices) * float(fraction)))) if float(fraction) > 0.0 else 0
	if count <= 0:
		return []
	selected = list(indices[-count:] if keep_from_end else indices[:count])
	return [int(index) for index in selected]


def _resolve_manual_split_section(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Resolve manual fire split config."""

	section = config.get("manual_fire_split")
	return dict(section) if isinstance(section, Mapping) else {}


def _resolve_artifact_path(config: Mapping[str, Any], configured_path: str | Path) -> Path:
	"""Resolve one artifact path relative to the config file when possible."""

	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	if config_path is None:
		return path.resolve()
	return (config_path.parent / path).resolve()


def _validate_manual_fire_lists(
	dataset_records: Sequence[Mapping[str, Any]],
	train_fire_names: Sequence[str],
	val_fire_names: Sequence[str],
	test_fire_names: Sequence[str],
	config: Mapping[str, Any],
) -> None:
	"""Validate manual split fire names and overlap constraints."""

	section = _resolve_manual_split_section(config)
	available = {str(record["dataset_name"]) for record in dataset_records}
	require_all = bool(section.get("require_all_listed_fires_exist", True))
	train_set = {str(name) for name in train_fire_names}
	val_set = {str(name) for name in val_fire_names}
	test_set = {str(name) for name in test_fire_names}
	listed = train_set | val_set | test_set

	if require_all:
		missing = sorted(name for name in listed if name not in available)
		if missing:
			raise ValueError(
				"manual_fire_split includes fire names that do not exist in the discovered dataset records: "
				f"{missing}"
			)

	if bool(section.get("disallow_overlap_between_splits", True)):
		overlaps = {
			"train/val": sorted(train_set & val_set),
			"train/test": sorted(train_set & test_set),
			"val/test": sorted(val_set & test_set),
		}
		active_overlaps = {name: values for name, values in overlaps.items() if values}
		if active_overlaps:
			raise ValueError(f"manual_fire_split contains overlapping fire assignments: {active_overlaps}")


def _print_manual_split_summary(
	rows: Sequence[tuple[str, str, str, int, int, int]],
	totals: Mapping[str, int],
) -> None:
	"""Print one manual split summary table."""

	print("Split | Fire name | HxW | files | valid temporal samples")
	for split_name, fire_name, spatial_text, file_count, sample_count, patches_per_sample in rows:
		if patches_per_sample > 0:
			print(
				f"{split_name.upper():<5} | {fire_name:<20} | {spatial_text:<11} | "
				f"{file_count:<5} | {sample_count:<21} | eval patches/sample={patches_per_sample}"
			)
		else:
			print(
				f"{split_name.upper():<5} | {fire_name:<20} | {spatial_text:<11} | "
				f"{file_count:<5} | {sample_count:<21}"
			)
	print(
		"Totals | "
		f"train fires={totals['train_fires']} val fires={totals['val_fires']} test fires={totals['test_fires']} | "
		f"train samples={totals['train_samples']} val samples={totals['val_samples']} test samples={totals['test_samples']}"
	)


def _maybe_warn_manual_fraction_drift(
	config: Mapping[str, Any],
	train_count: int,
	val_count: int,
	test_count: int,
) -> None:
	"""Warn when manual split sample counts drift far from requested fractions."""

	total = train_count + val_count + test_count
	if total <= 0:
		return
	expected = {
		"train": float(config.get("train_fraction", 0.0)),
		"val": float(config.get("val_fraction", 0.0)),
		"test": float(config.get("test_fraction", 0.0)),
	}
	actual = {
		"train": float(train_count / total),
		"val": float(val_count / total),
		"test": float(test_count / total),
	}
	tolerance = float(_resolve_manual_split_section(config).get("fraction_warning_tolerance", 0.20))
	drifts = {
		name: abs(actual[name] - expected[name])
		for name in ("train", "val", "test")
		if expected[name] > 0.0
	}
	if any(drift > tolerance for drift in drifts.values()):
		print(
			"WARNING: manual fire split sample proportions differ substantially from the configured "
			"train/val/test fractions. "
			f"expected={expected} actual={actual} tolerance={tolerance}"
		)


def _save_manual_split_json(
	dataset_records: Sequence[Mapping[str, Any]],
	config: Mapping[str, Any],
	split_refs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
	"""Persist the resolved manual split metadata when enabled."""

	section = _resolve_manual_split_section(config)
	if not bool(section.get("save_resolved_split_json", True)):
		return
	output_path = _resolve_artifact_path(
		config,
		section.get("resolved_split_json", "artifacts/splits/manual_fire_split_resolved.json"),
	)
	records_by_name = {str(record["dataset_name"]): record for record in dataset_records}
	payload: dict[str, Any] = {
		"split_mode": "manual_fire_holdout",
		"created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
		"splits": {},
	}
	for split_name in ("train", "val", "test"):
		fire_names = sorted({str(ref["dataset_name"]) for ref in split_refs[split_name]})
		fires: list[dict[str, Any]] = []
		for fire_name in fire_names:
			record = records_by_name[fire_name]
			sample_count = sum(1 for ref in split_refs[split_name] if str(ref["dataset_name"]) == fire_name)
			fires.append(
				{
					"fire_name": fire_name,
					"data_dir": str(record["data_dir"]),
					"geom_path": str(record.get("geom_path")) if record.get("geom_path") is not None else None,
					"terrain_path": str(record.get("terrain_path")) if record.get("terrain_path") is not None else None,
					"num_files": int(record["num_files"]),
					"raw_shape": [int(value) for value in record["raw_shape"]],
					"valid_temporal_samples": int(sample_count),
				}
			)
		payload["splits"][split_name] = fires
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, sort_keys=True)
		handle.write("\n")


def manual_fire_holdout_splits(
	dataset_records: Sequence[Mapping[str, Any]],
	train_fire_names: Sequence[str],
	val_fire_names: Sequence[str],
	test_fire_names: Sequence[str],
	input_sequence_length: int,
	prediction_horizon: int,
	config: Mapping[str, Any],
) -> dict[str, list[dict[str, int | str]]]:
	"""Assign whole fires to train/val/test splits."""

	section = _resolve_manual_split_section(config)
	_validate_manual_fire_lists(dataset_records, train_fire_names, val_fire_names, test_fire_names, config)
	assignments = {
		"train": {str(name) for name in train_fire_names},
		"val": {str(name) for name in val_fire_names},
		"test": {str(name) for name in test_fire_names},
	}
	use_full_fire = bool(section.get("use_full_fire_for_split", True))
	fractions = {
		"train": float(section.get("within_fire_train_fraction", 1.0)),
		"val": float(section.get("within_fire_val_fraction", 1.0)),
		"test": float(section.get("within_fire_test_fraction", 1.0)),
	}

	split_refs: dict[str, list[dict[str, int | str]]] = {"train": [], "val": [], "test": []}
	rows: list[tuple[str, str, str, int, int, int]] = []
	patching = resolve_patching_config(config)
	for record in dataset_records:
		dataset_name = str(record["dataset_name"])
		split_name = next((name for name, names in assignments.items() if dataset_name in names), None)
		if split_name is None:
			continue
		num_timesteps = int(record["num_files"])
		valid_indices = _sample_starts_for_segment(
			segment_start=0,
			segment_end=num_timesteps,
			input_sequence_length=input_sequence_length,
			prediction_horizon=prediction_horizon,
		)
		if not use_full_fire:
			valid_indices = _take_fraction_of_indices(
				valid_indices,
				fractions[split_name],
				keep_from_end=(split_name == "test"),
			)
		for sample_index in valid_indices:
			split_refs[split_name].append(
				{
					"dataset_id": int(record["dataset_id"]),
					"dataset_name": dataset_name,
					"sample_index": int(sample_index),
					"fire_split_group": split_name,
				}
			)
		height, width = (int(value) for value in record["raw_shape"][:2])
		if patching["enabled"] and patching["eval_mode"] == "sliding_window":
			patches_per_sample = len(
				build_sliding_window_patches(
					height=height,
					width=width,
					patch_h=int(patching["eval_patch_size"]),
					patch_w=int(patching["eval_patch_size"]),
					stride_h=int(patching["eval_stride"]),
					stride_w=int(patching["eval_stride"]),
					include_border_patches=bool(patching["include_border_patches"]),
				)
			)
		else:
			patches_per_sample = 0
		rows.append(
			(
				split_name,
				dataset_name,
				f"{height}x{width}",
				int(record["num_files"]),
				int(len(valid_indices)),
				int(patches_per_sample),
			)
		)

	if bool(section.get("require_nonempty_train", True)) and not split_refs["train"]:
		raise ValueError("manual_fire_holdout produced an empty train split.")
	if bool(section.get("require_nonempty_val", True)) and not split_refs["val"]:
		raise ValueError("manual_fire_holdout produced an empty val split.")
	if bool(section.get("require_nonempty_test", True)) and not split_refs["test"]:
		raise ValueError("manual_fire_holdout produced an empty test split.")

	totals = {
		"train_fires": len(assignments["train"]),
		"val_fires": len(assignments["val"]),
		"test_fires": len(assignments["test"]),
		"train_samples": len(split_refs["train"]),
		"val_samples": len(split_refs["val"]),
		"test_samples": len(split_refs["test"]),
	}
	_print_manual_split_summary(rows, totals)
	_maybe_warn_manual_fraction_drift(
		config,
		train_count=len(split_refs["train"]),
		val_count=len(split_refs["val"]),
		test_count=len(split_refs["test"]),
	)
	_save_manual_split_json(dataset_records, config, split_refs)
	return split_refs


def build_eval_patch_refs(
	dataset_records: Sequence[Mapping[str, Any]],
	sample_refs: Sequence[Mapping[str, Any]],
	config: Mapping[str, Any],
	split_name: str | None = None,
) -> list[dict[str, Any]]:
	"""Expand temporal sample refs into deterministic patch refs."""

	patching = resolve_patching_config(config)
	patch_size = int(patching["eval_patch_size"])
	stride = int(patching["eval_stride"])
	records_by_id = {int(record["dataset_id"]): record for record in dataset_records}
	expanded: list[dict[str, Any]] = []
	summary_counts: dict[tuple[str, str], tuple[int, int]] = {}

	for ref in sample_refs:
		dataset_id = int(ref["dataset_id"])
		record = records_by_id[dataset_id]
		height, width = (int(value) for value in record["raw_shape"][:2])
		patches = build_sliding_window_patches(
			height=height,
			width=width,
			patch_h=patch_size,
			patch_w=patch_size,
			stride_h=stride,
			stride_w=stride,
			include_border_patches=bool(patching["include_border_patches"]),
		)
		key = (str(record["dataset_name"]), str(split_name or ref.get("fire_split_group", "")))
		summary_counts[key] = (summary_counts.get(key, (0, len(patches)))[0] + 1, len(patches))
		for patch in patches:
			expanded.append(
				{
					"dataset_id": dataset_id,
					"dataset_name": str(ref.get("dataset_name", record["dataset_name"])),
					"sample_index": int(ref["sample_index"]),
					"patch": dict(patch),
					"fire_split_group": str(ref.get("fire_split_group", split_name or "")),
				}
			)

	print("Split | Fire name | temporal samples | patches/sample | total patch samples")
	for (dataset_name, resolved_split_name), (temporal_samples, patches_per_sample) in sorted(summary_counts.items()):
		split_text = str(resolved_split_name or split_name or "").upper()
		total_patch_samples = int(temporal_samples * patches_per_sample)
		print(
			f"{split_text:<5} | {dataset_name:<20} | {temporal_samples:<16} | "
			f"{patches_per_sample:<14} | {total_patch_samples}"
		)
	return expanded


def chronological_split_indices(
	num_timesteps: int,
	input_sequence_length: int,
	prediction_horizon: int,
	train_fraction: float,
	val_fraction: float,
	test_fraction: float,
	split_mode: str = "train_val_test",
) -> Dict[str, List[int]]:
	"""Split forecasting sample indices chronologically.

	When ``split_mode == "train_val_external_test"``, the main dataset is split
	into train and validation only, and the returned internal test split is empty.
	"""

	split_mode = str(split_mode).lower()
	if split_mode == "train_val_external_test":
		train_val = chronological_train_val_split_indices(
			num_timesteps=num_timesteps,
			input_sequence_length=input_sequence_length,
			prediction_horizon=prediction_horizon,
			train_fraction=train_fraction,
			val_fraction=val_fraction,
		)
		return {"train": train_val["train"], "val": train_val["val"], "test": []}

	if num_timesteps <= 0:
		raise ValueError(f"num_timesteps must be positive, got {num_timesteps}.")
	if input_sequence_length <= 0:
		raise ValueError(f"input_sequence_length must be positive, got {input_sequence_length}.")
	if prediction_horizon < 0:
		raise ValueError(f"prediction_horizon must be non-negative, got {prediction_horizon}.")

	_validate_fractions(train_fraction, val_fraction, test_fraction)

	max_valid_start = num_timesteps - input_sequence_length - prediction_horizon
	if max_valid_start < 0:
		raise ValueError(
			"Not enough timesteps to form a single valid sample. "
			f"Need at least input_sequence_length + prediction_horizon = "
			f"{input_sequence_length + prediction_horizon}, got {num_timesteps}."
		)

	train_length = int(math.floor(num_timesteps * train_fraction))
	val_length = int(math.floor(num_timesteps * val_fraction))
	val_segment_start = train_length
	test_segment_start = train_length + val_length

	train = _sample_starts_for_segment(0, val_segment_start, input_sequence_length, prediction_horizon)
	val = _sample_starts_for_segment(val_segment_start, test_segment_start, input_sequence_length, prediction_horizon)
	test = _sample_starts_for_segment(test_segment_start, num_timesteps, input_sequence_length, prediction_horizon)
	return {"train": train, "val": val, "test": test}


if __name__ == "__main__":
	splits = chronological_train_val_split_indices(
		num_timesteps=100,
		input_sequence_length=6,
		prediction_horizon=1,
		train_fraction=0.85,
		val_fraction=0.15,
	)
	assert splits["train"], "train split should not be empty for the demo case"
	assert splits["val"], "val split should not be empty for the demo case"
	assert splits["train"] == sorted(splits["train"])
	assert splits["val"] == sorted(splits["val"])
	assert max(splits["train"]) < min(splits["val"])
	print("chronological_train_val_split_indices demo passed")
