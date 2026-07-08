"""Discovery and indexing helpers for fire dataset folders."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.geometry import parse_geom_file


DEFAULT_MAIN_DATA_DIR = Path("/media/mhabibp/Elements/Mobin_CPS_files/New_CAWFE/")


def _sorted_unique_directories(paths: Sequence[Path]) -> list[Path]:
	"""Return unique parent directories in sorted order."""

	return sorted({path.resolve() for path in paths})


def _discover_keepz_directories(fire_root: Path, recursive: bool) -> list[Path]:
	"""Find every keepz_08 directory beneath one fire root."""

	if recursive:
		keepz_dirs = [path for path in fire_root.rglob("keepz_08") if path.is_dir()]
	else:
		keepz_dirs = [path for path in fire_root.glob("keepz_08") if path.is_dir()]
	return _sorted_unique_directories(keepz_dirs)


def infer_raw_shape_from_first_npy(fire_dir: Path, file_pattern: str) -> list[int]:
	"""Load only the first `.npy` file and infer its shape."""

	files = sorted(fire_dir.glob(file_pattern))
	if not files:
		raise FileNotFoundError(f"No files found in {fire_dir} using pattern {file_pattern!r}.")
	first = np.load(files[0], mmap_mode="r", allow_pickle=False)
	return [int(dimension) for dimension in first.shape]


def _discover_candidate_npy_dirs(fire_root: Path, file_pattern: str, recursive: bool) -> list[Path]:
	"""Find candidate tensor directories beneath one fire root."""

	if recursive:
		npy_files = list(fire_root.rglob(file_pattern))
	else:
		npy_files = list(fire_root.glob(file_pattern))
		for child in sorted(path for path in fire_root.iterdir() if path.is_dir()):
			npy_files.extend(child.glob(file_pattern))
	candidate_dirs = _sorted_unique_directories([path.parent for path in npy_files if path.is_file()])
	keepz_dirs = [path for path in candidate_dirs if path.name == "keepz_08"]
	if keepz_dirs:
		return keepz_dirs
	return candidate_dirs


def _match_geom_to_shape(
	geom_files: Sequence[Path],
	height: int,
	width: int,
) -> list[tuple[Path, dict[str, Any], str]]:
	"""Return geom files whose declared dimensions match the tensor shape."""

	matches: list[tuple[Path, dict[str, Any], str]] = []
	for geom_path in geom_files:
		geom_info = parse_geom_file(geom_path)
		if int(geom_info["ny"]) == int(height) and int(geom_info["nx"]) == int(width):
			matches.append((geom_path, geom_info, "standard"))
		elif int(geom_info["nx"]) == int(height) and int(geom_info["ny"]) == int(width):
			matches.append((geom_path, geom_info, "transposed"))
	return matches


def _resolve_fire_key(fire_root: Path, npy_dirs: Sequence[Path], npy_dir: Path) -> str:
	"""Build a stable key for one discovered dataset entry."""

	relative_parts = npy_dir.resolve().relative_to(fire_root.resolve()).parts
	if len(npy_dirs) == 1 and len(relative_parts) == 1:
		return fire_root.name
	return "__".join((fire_root.name, *relative_parts))


def discover_fire_datasets(
	main_data_dir: Path,
	fire_dir_glob: str = "*",
	file_pattern: str = "*.npy",
	recursive: bool = True,
	require_npy_files: bool = True,
	require_geom: bool = True,
	require_terrain: bool = False,
) -> dict[str, Any]:
	"""Search a main dataset directory and build a per-fire dataset index."""

	main_data_dir = Path(main_data_dir).expanduser().resolve()
	if not main_data_dir.exists():
		raise FileNotFoundError(f"Main dataset directory does not exist: {main_data_dir}")
	if not main_data_dir.is_dir():
		raise NotADirectoryError(f"Main dataset path is not a directory: {main_data_dir}")

	fire_roots = sorted(path for path in main_data_dir.glob(fire_dir_glob) if path.is_dir())
	records: dict[str, Any] = {}

	for fire_root in fire_roots:
		npy_dirs = _discover_keepz_directories(fire_root, recursive=recursive)
		if not npy_dirs:
			npy_dirs = _discover_candidate_npy_dirs(fire_root, file_pattern=file_pattern, recursive=recursive)
		if not npy_dirs and require_npy_files:
			continue

		for npy_dir in npy_dirs:
			if recursive:
				npy_files = sorted(path for path in npy_dir.rglob(file_pattern) if path.is_file())
			else:
				npy_files = sorted(path for path in npy_dir.glob(file_pattern) if path.is_file())
			if require_npy_files and not npy_files:
				continue
			geom_files = sorted(path for path in npy_dir.glob("*.geom") if path.is_file())
			terrain_files = sorted(path for path in npy_dir.glob("*.terrain") if path.is_file())

			raw_shape = infer_raw_shape_from_first_npy(npy_dir, file_pattern=file_pattern)
			if len(raw_shape) < 2:
				raise ValueError(f"Expected tensor shape with at least 2 dimensions, got {raw_shape} in {npy_dir}")
			height, width = int(raw_shape[0]), int(raw_shape[1])

			matched_geoms = _match_geom_to_shape(geom_files, height=height, width=width)
			if require_geom and not matched_geoms:
				if geom_files:
					available_shapes = []
					for geom_path in geom_files:
						geom_info = parse_geom_file(geom_path)
						available_shapes.append(f"{geom_path.name}: (nx={geom_info['nx']}, ny={geom_info['ny']})")
					raise ValueError(
						f"No .geom file under {npy_dir} matched tensor shape (H={height}, W={width}). "
						"Expected either (ny, nx) or (nx, ny). "
						f"Available geom shapes: {available_shapes}"
					)
				raise FileNotFoundError(
					f"Missing .geom file for fire dataset: {npy_dir}. "
					"Energy release requires per-cell area from the .geom file."
				)
			if len(matched_geoms) > 1:
				raise ValueError(
					f"Multiple .geom files matched tensor shape {height}x{width} for fire dataset {npy_dir}: "
					f"{[str(path) for path, _ in matched_geoms]}"
				)

			geom_path = matched_geoms[0][0].resolve() if matched_geoms else None
			geom_info = matched_geoms[0][1] if matched_geoms else None
			geom_orientation = matched_geoms[0][2] if matched_geoms else None
			terrain_path = None
			if geom_path is not None:
				same_stem_terrain = geom_path.with_suffix(".terrain")
				if same_stem_terrain.exists():
					terrain_path = same_stem_terrain.resolve()
				elif len(terrain_files) == 1:
					terrain_path = terrain_files[0].resolve()
			if require_terrain and terrain_path is None:
				raise FileNotFoundError(f"Required .terrain file was not found for fire dataset: {npy_dir}")

			fire_key = _resolve_fire_key(fire_root, npy_dirs=npy_dirs, npy_dir=npy_dir)
			record = {
				"fire_name": fire_key,
				"fire_root_name": fire_root.name,
				"fire_root_dir": str(fire_root.resolve()),
				"data_dir": str(npy_dir.resolve()),
				"num_npy_files": int(len(npy_files)),
				"file_pattern": str(file_pattern),
				"geom_path": str(geom_path) if geom_path is not None else None,
				"terrain_path": str(terrain_path) if terrain_path is not None else None,
				"has_geom": bool(geom_path is not None),
				"has_terrain": bool(terrain_path is not None),
				"raw_shape": [int(value) for value in raw_shape],
				"geom_tensor_orientation": str(geom_orientation) if geom_orientation is not None else None,
				"geom_requires_transpose": bool(geom_orientation == "transposed"),
				"valid_for_energy_release": bool(geom_path is not None),
			}
			if geom_info is not None:
				record["nx_geom"] = int(geom_info["nx"])
				record["ny_geom"] = int(geom_info["ny"])
				record["nz_geom"] = int(geom_info["nz"])
			records[fire_key] = record

	if not records:
		raise FileNotFoundError(f"No valid fire datasets found under {main_data_dir}.")

	return {
		"main_data_dir": str(main_data_dir),
		"last_updated": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
		"num_fires": int(len(records)),
		"fires": records,
	}


def load_fire_dataset_index(output_json: Path) -> dict[str, Any]:
	"""Load a previously saved fire dataset index."""

	output_json = Path(output_json).expanduser().resolve()
	with output_json.open("r", encoding="utf-8") as handle:
		index = json.load(handle)
	if not isinstance(index, dict):
		raise ValueError(f"Expected fire dataset index JSON to contain an object, got {type(index)!r}.")
	return index


def update_fire_dataset_index(existing: Mapping[str, Any], discovered: Mapping[str, Any]) -> dict[str, Any]:
	"""Merge a newly discovered index into an existing one."""

	existing_fires = dict(existing.get("fires", {})) if isinstance(existing.get("fires"), Mapping) else {}
	discovered_fires = dict(discovered.get("fires", {})) if isinstance(discovered.get("fires"), Mapping) else {}
	merged_fires: dict[str, Any] = {}

	for fire_name, record in existing_fires.items():
		data_dir = Path(str(record.get("data_dir", ""))).expanduser()
		if data_dir.exists():
			merged_fires[str(fire_name)] = dict(record)

	for fire_name, record in discovered_fires.items():
		merged_fires[str(fire_name)] = dict(record)

	merged = dict(discovered)
	merged["fires"] = merged_fires
	merged["num_fires"] = int(len(merged_fires))
	return merged


def save_fire_dataset_index(index: Mapping[str, Any], output_json: Path) -> None:
	"""Save the fire dataset index as JSON."""

	output_json = Path(output_json).expanduser().resolve()
	output_json.parent.mkdir(parents=True, exist_ok=True)
	with output_json.open("w", encoding="utf-8") as handle:
		json.dump(index, handle, indent=2, sort_keys=True)
		handle.write("\n")
