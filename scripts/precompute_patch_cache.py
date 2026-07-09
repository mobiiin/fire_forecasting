"""Precompute training-ready wildfire patch shards on scratch storage."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional .pt shards
	torch = None

from src.config import load_config
from src.data.cache import MANIFEST_FILENAME, compute_cache_config_hash, get_patch_cache_dir, load_cache_manifest
from src.data.dataset import MultiFirePatchSequenceDataset
from src.data.discovery import discover_multiple_datasets
from src.data.energy_release import resolve_energy_output_channel_names, resolve_energy_release_config
from src.data.patching import resolve_patching_config
from src.data.splits import build_eval_patch_refs, manual_fire_holdout_splits, multi_fire_chronological_splits


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


def _selected_splits(split: str) -> list[str]:
	split = str(split).lower()
	if split == "all":
		return ["train", "val", "test"]
	if split not in {"train", "val", "test"}:
		raise ValueError(f"split must be train, val, test, or all. Got {split!r}.")
	return [split]


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
	if bool(patching["enabled"]) and str(patching["eval_mode"]) == "sliding_window":
		sample_refs = {
			"train": list(sample_refs["train"]),
			"val": build_eval_patch_refs(dataset_records=dataset_records, sample_refs=sample_refs["val"], config=config, split_name="val"),
			"test": build_eval_patch_refs(dataset_records=dataset_records, sample_refs=sample_refs["test"], config=config, split_name="test"),
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
	return {
		"cache_version": str(cache_config.get("cache_version", "v1")),
		"created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
		"config_hash": compute_cache_config_hash(config),
		"split_mode": str(config.get("split_mode", "train_val_test")),
		"manual_fire_split": _get_section(config, "manual_fire_split"),
		"input_sequence_length": int(config["input_sequence_length"]),
		"prediction_horizon": int(config["prediction_horizon"]),
		"input_channels": int(dataset.total_input_channels),
		"base_input_channel_count": int(dataset.base_input_channel_count),
		"fuel_flux_engineered_channel_count": int(dataset.fuel_flux_engineered_channel_count),
		"atmospheric_engineered_channel_count": int(dataset.atmospheric_engineered_channel_count),
		"energy_history_channel_count": int(dataset.energy_history_channel_count),
		"engineered_channel_count": int(dataset.engineered_channel_count),
		"output_channels": int(_get_section(config, "model").get("output_channels", 1)),
		"patch_height": int(patching["patch_height"]),
		"patch_width": int(patching["patch_width"]),
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
	row = {
		"split": split,
		"fire_name": str(metadata.get("dataset_name", metadata.get("data_dir", ""))),
		"dataset_id": int(metadata.get("dataset_id", -1)),
		"sample_index": int(metadata.get("sample_index", -1)),
		"current_idx": int(metadata.get("current_idx", metadata.get("current_index", -1))),
		"future_idx": int(metadata.get("future_idx", metadata.get("future_index", -1))),
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
	metadata_handle,
) -> dict[str, Any]:
	shard_format = str(cache_config.get("shard_format", "npz")).lower()
	compressed = bool(cache_config.get("compressed", False))
	extension = ".pt" if shard_format == "pt" else ".npz"
	shard_name = f"shard_{shard_index:06d}{extension}"
	shard_path = split_dir / shard_name
	x_array = np.stack(x_items, axis=0).astype(np.float32, copy=False)
	y_array = np.stack(y_items, axis=0).astype(np.float32, copy=False)
	sample_ids = np.asarray([str(item["sample_id"]) for item in metadata_items])
	dataset_ids = np.asarray([int(item["dataset_id"]) for item in metadata_items], dtype=np.int64)
	patch_y0 = np.asarray([int(item["patch"]["y0"]) for item in metadata_items], dtype=np.int64)
	patch_x0 = np.asarray([int(item["patch"]["x0"]) for item in metadata_items], dtype=np.int64)
	sample_indices = np.asarray([int(item["sample_index"]) for item in metadata_items], dtype=np.int64)

	if shard_format == "npz":
		save_fn = np.savez_compressed if compressed else np.savez
		save_fn(
			shard_path,
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
			shard_path,
		)
	else:
		raise ValueError(f"Unsupported cache.shard_format={shard_format!r}. Expected 'npz' or 'pt'.")

	for local_index, item in enumerate(metadata_items):
		item = dict(item)
		item["shard"] = f"{split}/{shard_name}"
		item["local_index"] = int(local_index)
		metadata_handle.write(json.dumps(item, sort_keys=True) + "\n")

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
		"current_idx",
		"future_idx",
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
						"current_idx": item.get("current_idx"),
						"future_idx": item.get("future_idx"),
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
) -> None:
	cache_config = _get_section(config, "cache")
	samples_per_shard = int(cache_config.get("samples_per_shard", 512))
	if samples_per_shard <= 0:
		raise ValueError(f"cache.samples_per_shard must be positive, got {samples_per_shard}.")
	split_dir = cache_dir / split
	split_dir.mkdir(parents=True, exist_ok=True)
	metadata_path = split_dir / "metadata.jsonl"
	preview_enabled = bool(cache_config.get("save_preview_images", True))
	max_previews = int(cache_config.get("num_preview_images", 50))
	save_metadata = bool(cache_config.get("save_metadata", True))

	shards: list[dict[str, Any]] = []
	x_buffer: list[np.ndarray] = []
	y_buffer: list[np.ndarray] = []
	metadata_buffer: list[dict[str, Any]] = []
	preview_count = 0
	shard_index = 0
	total_samples = 0
	warned_large_shard = False
	metadata_handle = metadata_path.open("w", encoding="utf-8") if save_metadata else open(Path("/dev/null"), "w", encoding="utf-8")
	try:
		for item_index in range(len(dataset)):
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
				shards.append(_write_shard(split_dir, split, shard_index, x_buffer, y_buffer, metadata_buffer, cache_config, metadata_handle))
				shard_index += 1
				x_buffer.clear()
				y_buffer.clear()
				metadata_buffer.clear()
			if total_samples % 100 == 0:
				print(f"{split}: cached {total_samples}/{len(dataset)} patches")

		if x_buffer:
			shards.append(_write_shard(split_dir, split, shard_index, x_buffer, y_buffer, metadata_buffer, cache_config, metadata_handle))
	finally:
		metadata_handle.close()

	manifest["shards"][split] = shards
	manifest[f"num_{split}_patches"] = int(total_samples)
	print(f"{split}: wrote {total_samples} patches across {len(shards)} shard(s) to {split_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Precompute wildfire ConvLSTM patch-cache shards.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML configuration file.")
	parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"], help="Split to precompute.")
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing shard files for the selected split(s).")
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
	overwrite = bool(args.overwrite or cache_config.get("overwrite_existing", False))
	cache_dir.mkdir(parents=True, exist_ok=True)

	for split in selected:
		split_dir = cache_dir / split
		if split_dir.exists() and any(split_dir.glob("shard_*")):
			if not overwrite:
				raise FileExistsError(
					f"Patch-cache split directory already contains shards: {split_dir}. "
					"Set cache.overwrite_existing=true or pass --overwrite to replace it."
				)
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
	if manifest_path.exists() and (not overwrite or not rebuilding_all_splits):
		manifest = load_cache_manifest(cache_dir)
		current_hash = compute_cache_config_hash(config)
		if str(manifest.get("config_hash")) != current_hash and not overwrite:
			raise RuntimeError(
				f"Existing manifest config hash differs from current config under {cache_dir}. "
				"Pass --overwrite if you intend to rebuild the cache."
			)
	else:
		manifest = _base_manifest(config, dataset_records, split_refs, first_dataset)

	for split in selected:
		dataset = first_dataset if split == selected[0] else _build_dataset(
			config=config,
			dataset_records=dataset_records,
			sample_refs=split_refs[split],
			split=split,
			normalization_stats=normalization_stats,
		)
		_precompute_split(config, dataset, split, cache_dir, manifest)

	manifest["created_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
	manifest["config_hash"] = compute_cache_config_hash(config)
	with manifest_path.open("w", encoding="utf-8") as handle:
		json.dump(manifest, handle, indent=2, sort_keys=True)
		handle.write("\n")
	_rebuild_summary_csv(cache_dir)
	print(f"Patch cache manifest: {manifest_path}")
	print(f"Patch summary CSV: {cache_dir / 'patch_summary.csv'}")


if __name__ == "__main__":
	main()
