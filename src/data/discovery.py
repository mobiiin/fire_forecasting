"""Dataset discovery helpers for single- and multi-directory wildfire datasets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _extract_numeric_suffix(name: str) -> int | None:
	"""Extract a trailing numeric suffix from a filename stem when present."""

	match = re.search(r"(\d+)$", name)
	return int(match.group(1)) if match else None


def sort_chronologically(file_paths: Sequence[Path]) -> list[Path]:
	"""Sort files by trailing numeric suffix when available, otherwise lexicographically."""

	numeric_suffixes = [_extract_numeric_suffix(path.stem) for path in file_paths]
	if all(value is not None for value in numeric_suffixes):
		return [path for _, path in sorted(zip(numeric_suffixes, file_paths), key=lambda item: item[0])]
	return sorted(file_paths, key=lambda path: path.name)


def _resolve_path(base_path: Path | None, configured_path: str | Path) -> Path:
	"""Resolve a configured path relative to a config file when needed."""

	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	if base_path is None:
		return path.resolve()
	return (base_path.parent / path).resolve()


def discover_dataset_files(data_dir: Path, file_pattern: str) -> list[Path]:
	"""Discover and chronologically sort one dataset directory."""

	files = sort_chronologically(list(data_dir.glob(str(file_pattern))))
	if not files:
		raise FileNotFoundError(f"No files found in '{data_dir}' using pattern '{file_pattern}'.")
	return files


def resolve_data_dirs(config: Mapping[str, Any]) -> list[Path]:
	"""Resolve configured dataset directories, preferring ``data_dirs`` over ``data_dir``."""

	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	configured_data_dirs = config.get("data_dirs")
	if isinstance(configured_data_dirs, Sequence) and not isinstance(configured_data_dirs, (str, bytes)):
		resolved = [_resolve_path(config_path, item) for item in configured_data_dirs if str(item).strip()]
		if resolved:
			return resolved

	configured_data_dir = config.get("data_dir")
	if configured_data_dir in (None, "", "null"):
		raise KeyError("Config must define either a non-empty data_dirs list or a legacy data_dir path.")
	return [_resolve_path(config_path, configured_data_dir)]


def discover_multiple_datasets(config: Mapping[str, Any]) -> list[dict[str, Any]]:
	"""Discover one or more dataset directories and validate basic consistency."""

	file_pattern = str(config.get("file_pattern", "*.npy"))
	if "file_pattern" not in config:
		raise KeyError("Config is missing required key 'file_pattern'.")

	use_patches = bool(config.get("use_patches", False))
	patch_size = int(config.get("patch_size", 64))
	data_dirs = resolve_data_dirs(config)
	dataset_records: list[dict[str, Any]] = []
	reference_channel_count: int | None = None
	reference_shape_hw: tuple[int, int] | None = None
	spatial_sizes: set[tuple[int, int]] = set()

	for dataset_id, data_dir in enumerate(data_dirs):
		if not data_dir.exists():
			raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
		file_paths = discover_dataset_files(data_dir, file_pattern)
		first_tensor = np.load(file_paths[0], mmap_mode="r", allow_pickle=False)
		if first_tensor.ndim != 3:
			raise ValueError(
				f"Expected dataset files to contain 3D tensors, got shape {first_tensor.shape} in {file_paths[0]}."
			)
		raw_shape = tuple(int(dimension) for dimension in first_tensor.shape)
		height, width, channels = raw_shape
		if reference_channel_count is None:
			reference_channel_count = channels
		elif channels != reference_channel_count:
			raise ValueError(
				"All datasets must have the same raw channel count. "
				f"Expected {reference_channel_count}, got {channels} in {data_dir}."
			)
		if patch_size > min(height, width) and use_patches:
			raise ValueError(
				"patch_size must fit inside every dataset when use_patches=true. "
				f"Got patch_size={patch_size}, dataset={data_dir.name}, raw_shape={raw_shape}."
			)
		reference_shape_hw = reference_shape_hw or (height, width)
		spatial_sizes.add((height, width))
		dataset_records.append(
			{
				"dataset_id": int(dataset_id),
				"dataset_name": str(data_dir.name),
				"data_dir": data_dir,
				"file_paths": file_paths,
				"num_files": len(file_paths),
				"raw_shape": raw_shape,
			}
		)

	if not dataset_records:
		raise ValueError("No datasets were discovered from the configured data directories.")

	if len(spatial_sizes) > 1:
		print(f"WARNING: discovered mixed raw spatial sizes across datasets: {sorted(spatial_sizes)}")
		if not use_patches:
			raise ValueError(
				"Mixed spatial sizes were discovered across data_dirs, but use_patches=false. "
				"Enable patch mode or make spatial sizes consistent before batching multiple datasets together."
			)

	return dataset_records
