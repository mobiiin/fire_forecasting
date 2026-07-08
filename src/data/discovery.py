"""Dataset discovery helpers for single- and multi-directory wildfire datasets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.fire_index import (
	DEFAULT_MAIN_DATA_DIR,
	discover_fire_datasets,
	load_fire_dataset_index,
	update_fire_dataset_index,
	save_fire_dataset_index,
)
from src.data.geometry import load_fire_geometry


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


def _resolve_discovery_config(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Resolve top-level dataset discovery settings."""

	data_discovery = config.get("data_discovery", {}) if isinstance(config.get("data_discovery"), Mapping) else {}
	return {
		"mode": str(data_discovery.get("mode", "explicit_data_dirs")).lower(),
		"fire_dir_glob": str(data_discovery.get("fire_dir_glob", "*")),
		"require_npy_files": bool(data_discovery.get("require_npy_files", True)),
		"file_pattern": str(data_discovery.get("file_pattern", config.get("file_pattern", "*.npy"))),
		"require_geom": bool(data_discovery.get("require_geom", True)),
		"require_terrain": bool(data_discovery.get("require_terrain", False)),
		"recursive": bool(data_discovery.get("recursive", True)),
		"update_fire_index_before_training": bool(data_discovery.get("update_fire_index_before_training", False)),
	}


def _resolve_fire_filter_config(config: Mapping[str, Any]) -> dict[str, list[str]]:
	"""Resolve optional include/exclude fire filters."""

	section = config.get("fire_filter", {}) if isinstance(config.get("fire_filter"), Mapping) else {}
	def _coerce_list(key: str) -> list[str]:
		value = section.get(key, [])
		if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
			return [str(item) for item in value if str(item).strip()]
		return []
	return {
		"include_fires": _coerce_list("include_fires"),
		"exclude_fires": _coerce_list("exclude_fires"),
	}


def _resolve_manual_split_selected_fires(config: Mapping[str, Any]) -> set[str] | None:
	"""Return the explicitly selected fires when manual holdout is active."""

	split_mode = str(config.get("split_mode", "")).lower()
	if split_mode != "manual_fire_holdout":
		return None
	section = config.get("manual_fire_split", {}) if isinstance(config.get("manual_fire_split"), Mapping) else {}
	selected: set[str] = set()
	for key in ("train_fires", "val_fires", "test_fires"):
		value = section.get(key, [])
		if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
			selected.update(str(item) for item in value if str(item).strip())
	return selected


def _apply_data_dir_record_filters(data_dir_records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
	"""Apply include/exclude fire filters and manual split selection."""

	selected_manual = _resolve_manual_split_selected_fires(config)
	fire_filter = _resolve_fire_filter_config(config)
	include_fires = set(fire_filter["include_fires"])
	exclude_fires = set(fire_filter["exclude_fires"])
	filtered: list[dict[str, Any]] = []

	for record in data_dir_records:
		dataset_name = str(record["dataset_name"])
		if selected_manual is not None:
			if dataset_name not in selected_manual:
				continue
		elif include_fires and dataset_name not in include_fires:
			continue
		if selected_manual is None and dataset_name in exclude_fires:
			continue
		filtered.append(dict(record))
	return filtered


def _resolve_main_data_dir(config: Mapping[str, Any]) -> Path:
	"""Resolve the configured main dataset directory."""

	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	return _resolve_path(config_path, config.get("main_data_dir", DEFAULT_MAIN_DATA_DIR))


def _default_fire_index_json_path() -> Path:
	"""Return the project-root default fire-index path."""

	return Path(__file__).resolve().parents[2] / "fire_dataset_index.json"


def _resolve_fire_index_json_path(config: Mapping[str, Any]) -> Path:
	"""Resolve the fire dataset index JSON path."""

	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	default_path = _default_fire_index_json_path()
	return _resolve_path(config_path, config.get("fire_dataset_index_json", default_path))


def _load_or_refresh_fire_index(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Load the configured fire index, optionally refreshing it first."""

	discovery = _resolve_discovery_config(config)
	main_data_dir = _resolve_main_data_dir(config)
	index_path = _resolve_fire_index_json_path(config)
	if discovery["update_fire_index_before_training"]:
		discovered = discover_fire_datasets(
			main_data_dir=main_data_dir,
			fire_dir_glob=discovery["fire_dir_glob"],
			file_pattern=discovery["file_pattern"],
			recursive=discovery["recursive"],
			require_npy_files=discovery["require_npy_files"],
			require_geom=discovery["require_geom"],
			require_terrain=discovery["require_terrain"],
		)
		if index_path.exists():
			discovered = update_fire_dataset_index(load_fire_dataset_index(index_path), discovered)
		save_fire_dataset_index(discovered, index_path)
		return discovered
	if not index_path.exists():
		raise FileNotFoundError(
			"fire_dataset_index.json not found. Run:\n"
			f"python scripts/discover_fire_datasets.py --main_data_dir {main_data_dir}"
		)
	return load_fire_dataset_index(index_path)


def _records_from_fire_index(config: Mapping[str, Any]) -> list[dict[str, Any]]:
	"""Build dataset records from a saved fire index."""

	index = _load_or_refresh_fire_index(config)
	fires = index.get("fires", {})
	if not isinstance(fires, Mapping):
		raise ValueError("fire_dataset_index.json is malformed: expected a top-level 'fires' mapping.")
	records: list[dict[str, Any]] = []
	for fire_name, record in sorted(fires.items()):
		if not isinstance(record, Mapping):
			continue
		if not bool(record.get("valid_for_energy_release", False)):
			continue
		records.append(
			{
				"dataset_name": str(record.get("fire_name", fire_name)),
				"data_dir": Path(str(record["data_dir"])).expanduser().resolve(),
				"geom_path": Path(str(record["geom_path"])).expanduser().resolve() if record.get("geom_path") else None,
				"terrain_path": Path(str(record["terrain_path"])).expanduser().resolve() if record.get("terrain_path") else None,
				"fire_root_dir": Path(str(record.get("fire_root_dir", record["data_dir"]))).expanduser().resolve(),
				"geom_requires_transpose": bool(record.get("geom_requires_transpose", False)),
				"geom_tensor_orientation": record.get("geom_tensor_orientation"),
			}
		)
	return records


def _index_record_lookup_by_data_dir(index: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
	"""Build a lookup from resolved data_dir string to fire-index record."""

	fires = index.get("fires", {})
	if not isinstance(fires, Mapping):
		return {}
	lookup: dict[str, Mapping[str, Any]] = {}
	for record in fires.values():
		if isinstance(record, Mapping) and record.get("data_dir"):
			lookup[str(Path(str(record["data_dir"])).expanduser().resolve())] = record
	return lookup


def _records_from_scan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
	"""Build dataset records by scanning the main dataset directory directly."""

	discovery = _resolve_discovery_config(config)
	main_data_dir = _resolve_main_data_dir(config)
	index = discover_fire_datasets(
		main_data_dir=main_data_dir,
		fire_dir_glob=discovery["fire_dir_glob"],
		file_pattern=discovery["file_pattern"],
		recursive=discovery["recursive"],
		require_npy_files=discovery["require_npy_files"],
		require_geom=discovery["require_geom"],
		require_terrain=discovery["require_terrain"],
	)
	records: list[dict[str, Any]] = []
	for fire_name, record in sorted(index["fires"].items()):
		records.append(
			{
				"dataset_name": str(record.get("fire_name", fire_name)),
				"data_dir": Path(str(record["data_dir"])).expanduser().resolve(),
				"geom_path": Path(str(record["geom_path"])).expanduser().resolve() if record.get("geom_path") else None,
				"terrain_path": Path(str(record["terrain_path"])).expanduser().resolve() if record.get("terrain_path") else None,
				"fire_root_dir": Path(str(record.get("fire_root_dir", record["data_dir"]))).expanduser().resolve(),
				"geom_requires_transpose": bool(record.get("geom_requires_transpose", False)),
				"geom_tensor_orientation": record.get("geom_tensor_orientation"),
			}
		)
	return records


def discover_data_dir_records(config: Mapping[str, Any]) -> list[dict[str, Any]]:
	"""Resolve dataset directories plus optional geometry metadata."""

	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	configured_data_dirs = config.get("data_dirs")
	if isinstance(configured_data_dirs, Sequence) and not isinstance(configured_data_dirs, (str, bytes)):
		index_lookup: dict[str, Mapping[str, Any]] = {}
		index_path = _resolve_fire_index_json_path(config)
		if index_path.exists():
			index_lookup = _index_record_lookup_by_data_dir(load_fire_dataset_index(index_path))
		resolved = []
		for item in configured_data_dirs:
			if str(item).strip():
				data_dir = _resolve_path(config_path, item)
				index_record = index_lookup.get(str(data_dir.resolve()))
				resolved.append(
					{
						"dataset_name": str(index_record.get("fire_name", data_dir.name)) if index_record is not None else data_dir.name,
						"data_dir": data_dir,
						"geom_path": Path(str(index_record["geom_path"])).expanduser().resolve() if index_record is not None and index_record.get("geom_path") else None,
						"terrain_path": Path(str(index_record["terrain_path"])).expanduser().resolve() if index_record is not None and index_record.get("terrain_path") else None,
						"geom_requires_transpose": bool(index_record.get("geom_requires_transpose", False)) if index_record is not None else False,
						"geom_tensor_orientation": index_record.get("geom_tensor_orientation") if index_record is not None else None,
					}
				)
		if resolved:
			return resolved

	discovery = _resolve_discovery_config(config)
	mode = discovery["mode"]
	if mode == "fire_index":
		return _records_from_fire_index(config)
	if mode == "scan_main_data_dir":
		return _records_from_scan(config)

	configured_data_dir = config.get("data_dir")
	if configured_data_dir in (None, "", "null"):
		raise KeyError("Config must define either a non-empty data_dirs list or a legacy data_dir path.")
	data_dir = _resolve_path(config_path, configured_data_dir)
	return [{"dataset_name": data_dir.name, "data_dir": data_dir, "geom_path": None, "terrain_path": None}]


def resolve_data_dirs(config: Mapping[str, Any]) -> list[Path]:
	"""Resolve configured dataset directories after applying discovery mode."""

	return [Path(record["data_dir"]).resolve() for record in discover_data_dir_records(config)]


def discover_multiple_datasets(config: Mapping[str, Any]) -> list[dict[str, Any]]:
	"""Discover one or more dataset directories and validate basic consistency."""

	file_pattern = str(config.get("file_pattern", "*.npy"))
	if "file_pattern" not in config:
		raise KeyError("Config is missing required key 'file_pattern'.")

	use_patches = bool(config.get("use_patches", False))
	patch_size = int(config.get("patch_size", 64))
	data_dir_records = _apply_data_dir_record_filters(discover_data_dir_records(config), config)
	if not data_dir_records:
		raise ValueError("No dataset directories remained after applying discovery/manual fire filters.")
	dataset_records: list[dict[str, Any]] = []
	reference_channel_count: int | None = None
	spatial_sizes: set[tuple[int, int]] = set()
	energy_enabled = bool(config.get("energy_release", {}).get("enabled", False)) if isinstance(config.get("energy_release"), Mapping) else False

	for dataset_id, source_record in enumerate(data_dir_records):
		data_dir = Path(source_record["data_dir"]).resolve()
		if not data_dir.exists():
			raise FileNotFoundError(
				f"Data directory does not exist for discovered fire {source_record.get('dataset_name', data_dir.name)!r}: {data_dir}. "
				"If this path came from fire_dataset_index.json, refresh the index with "
				"'python scripts/discover_fire_datasets.py --main_data_dir /media/mhabibp/Elements/Mobin_CPS_files/New_CAWFE/'."
			)
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
		spatial_sizes.add((height, width))
		record = {
			"dataset_id": int(dataset_id),
			"dataset_name": str(source_record.get("dataset_name", data_dir.name)),
			"data_dir": data_dir,
			"file_paths": file_paths,
			"num_files": len(file_paths),
			"raw_shape": raw_shape,
			"geom_requires_transpose": bool(source_record.get("geom_requires_transpose", False)),
			"geom_tensor_orientation": source_record.get("geom_tensor_orientation"),
		}
		if energy_enabled:
			geometry_config = dict(config)
			if bool(source_record.get("geom_requires_transpose", False)):
				geometry_section = dict(geometry_config.get("geometry", {})) if isinstance(geometry_config.get("geometry"), Mapping) else {}
				geometry_section["allow_area_transpose_if_needed"] = True
				geometry_config["geometry"] = geometry_section
			record["geometry"] = load_fire_geometry(
				data_dir=data_dir,
				config=geometry_config,
				geom_path=source_record.get("geom_path"),
				terrain_path=source_record.get("terrain_path"),
				expected_shape=(height, width),
			)
		dataset_records.append(record)

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
