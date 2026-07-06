"""Geometry helpers for per-cell CAWFE area maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _get_section(config: Mapping[str, Any] | None, *names: str) -> dict[str, Any]:
	"""Return the first nested mapping found under any of the provided names."""

	if not isinstance(config, Mapping):
		return {}
	for name in names:
		section = config.get(name)
		if isinstance(section, Mapping):
			return dict(section)
	return {}


def resolve_geometry_config(config: Mapping[str, Any]) -> dict[str, Any]:
	"""Resolve geometry config with defaults."""

	section = _get_section(config, "geometry")
	return {
		"require_geom_file": bool(section.get("require_geom_file", True)),
		"earth_radius_m": float(section.get("earth_radius_m", 6_371_000.0)),
		"use_geom_cos_latitude": bool(section.get("use_geom_cos_latitude", False)),
		"geom_cos_tolerance": float(section.get("geom_cos_tolerance", 1.0e-5)),
		"spacing_tolerance_relative": float(section.get("spacing_tolerance_relative", 1.0e-4)),
		"validate_against_terrain_header": bool(section.get("validate_against_terrain_header", True)),
		"allow_area_transpose_if_needed": bool(section.get("allow_area_transpose_if_needed", False)),
	}


def find_geom_file(data_dir: Path) -> Path:
	"""Find exactly one `.geom` file in a dataset directory."""

	geom_files = sorted(path for path in data_dir.glob("*.geom") if path.is_file())
	if not geom_files:
		raise FileNotFoundError(
			f"Missing .geom file for fire dataset: {data_dir}. Energy release requires per-cell area from the .geom file."
		)
	if len(geom_files) > 1:
		raise ValueError(
			f"Multiple .geom files found in fire dataset directory {data_dir}: {[path.name for path in geom_files]}"
		)
	print(f"Selected geom file: {geom_files[0]}")
	return geom_files[0]


def find_terrain_file(data_dir: Path) -> Path | None:
	"""Find a `.terrain` file in a dataset directory when present."""

	terrain_files = sorted(path for path in data_dir.glob("*.terrain") if path.is_file())
	if not terrain_files:
		return None
	if len(terrain_files) > 1:
		raise ValueError(
			f"Multiple .terrain files found in fire dataset directory {data_dir}: {[path.name for path in terrain_files]}"
		)
	return terrain_files[0]


def parse_geom_file(geom_path: Path) -> dict[str, Any]:
	"""Parse a `.geom` file into longitude, latitude, and cosine arrays."""

	content = geom_path.read_text(encoding="utf-8", errors="ignore").split()
	if len(content) < 3:
		raise ValueError(f"Geom file is too short to contain nx/ny/nz header: {geom_path}")

	try:
		nx = int(float(content[0]))
		ny = int(float(content[1]))
		nz = int(float(content[2]))
	except ValueError as exc:
		raise ValueError(f"Could not parse nx/ny/nz from geom file: {geom_path}") from exc

	expected_token_count = 3 + nx + ny + ny
	if len(content) < expected_token_count:
		raise ValueError(
			f"Geom file {geom_path} does not contain enough values. "
			f"Expected at least {expected_token_count}, found {len(content)}."
		)

	values = np.asarray([float(token) for token in content[3:expected_token_count]], dtype=np.float64)
	lons = values[:nx]
	lats = values[nx : nx + ny]
	cos_lats_from_file = values[nx + ny : nx + ny + ny]
	return {
		"geom_path": geom_path,
		"nx": int(nx),
		"ny": int(ny),
		"nz": int(nz),
		"lons": lons.astype(np.float64, copy=False),
		"lats": lats.astype(np.float64, copy=False),
		"cos_lats_from_file": cos_lats_from_file.astype(np.float64, copy=False),
	}


def parse_terrain_header(terrain_path: Path) -> dict[str, Any]:
	"""Parse the first-line header of a `.terrain` file."""

	first_line = terrain_path.read_text(encoding="utf-8", errors="ignore").splitlines()
	if not first_line:
		raise ValueError(f"Terrain file is empty: {terrain_path}")
	parts = first_line[0].split()
	if len(parts) < 5:
		raise ValueError(
			f"Terrain header in {terrain_path} must contain at least 5 values `nx ny nz dx dy`, got {parts!r}."
		)
	return {
		"nx": int(float(parts[0])),
		"ny": int(float(parts[1])),
		"nz": int(float(parts[2])),
		"dx_header_m": float(parts[3]),
		"dy_header_m": float(parts[4]),
	}


def compute_cell_area_from_geom(geom_info: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
	"""Compute a 2D per-cell area map from parsed geom coordinates."""

	geometry = resolve_geometry_config(config)
	nx = int(geom_info["nx"])
	ny = int(geom_info["ny"])
	if nx < 2 or ny < 2:
		raise ValueError(f"Geom grids must have at least 2 points in x and y, got nx={nx}, ny={ny}.")

	lons = np.asarray(geom_info["lons"], dtype=np.float64)
	lats = np.asarray(geom_info["lats"], dtype=np.float64)
	cos_lats_from_file = np.asarray(geom_info["cos_lats_from_file"], dtype=np.float64)

	lon_diffs_deg = np.abs(np.diff(lons))
	lat_diffs_deg = np.abs(np.diff(lats))
	median_dlon_deg = float(np.median(lon_diffs_deg))
	median_dlat_deg = float(np.median(lat_diffs_deg))
	if median_dlon_deg <= 0.0 or median_dlat_deg <= 0.0:
		raise ValueError("Longitude and latitude spacing from geom must be positive.")

	spacing_tolerance_relative = float(geometry["spacing_tolerance_relative"])
	if np.max(np.abs(lon_diffs_deg - median_dlon_deg)) > spacing_tolerance_relative * median_dlon_deg:
		print(f"WARNING: non-uniform longitude spacing detected in {geom_info['geom_path']}; using median spacing.")
	if np.max(np.abs(lat_diffs_deg - median_dlat_deg)) > spacing_tolerance_relative * median_dlat_deg:
		print(f"WARNING: non-uniform latitude spacing detected in {geom_info['geom_path']}; using median spacing.")

	computed_cos_lats = np.cos(np.deg2rad(lats))
	cos_tolerance = float(geometry["geom_cos_tolerance"])
	max_cos_error = float(np.max(np.abs(cos_lats_from_file - computed_cos_lats)))
	if max_cos_error > cos_tolerance:
		print(
			f"WARNING: cos(latitude) values in {geom_info['geom_path']} differ from cos(lats) by up to {max_cos_error:.6g}. "
			"Using values computed from lats unless geometry.use_geom_cos_latitude=true."
		)

	cos_lats = cos_lats_from_file if geometry["use_geom_cos_latitude"] else computed_cos_lats
	earth_radius_m = float(geometry["earth_radius_m"])
	dlon_rad = float(np.deg2rad(median_dlon_deg))
	dlat_rad = float(np.deg2rad(median_dlat_deg))
	dy_m = float(earth_radius_m * dlat_rad)
	dx_by_row_m = (earth_radius_m * cos_lats * dlon_rad).astype(np.float64, copy=False)
	area_by_row_m2 = (dx_by_row_m * dy_m).astype(np.float64, copy=False)
	area_2d_m2 = np.repeat(area_by_row_m2[:, None], nx, axis=1).astype(np.float32, copy=False)

	return {
		"area_2d_m2": area_2d_m2,
		"dx_by_row_m": dx_by_row_m.astype(np.float32, copy=False),
		"dy_m": float(dy_m),
		"area_by_row_m2": area_by_row_m2.astype(np.float32, copy=False),
		"dlon_rad": float(dlon_rad),
		"dlat_rad": float(dlat_rad),
		"earth_radius_m": float(earth_radius_m),
		"area_min_m2": float(np.min(area_2d_m2)),
		"area_max_m2": float(np.max(area_2d_m2)),
		"area_mean_m2": float(np.mean(area_2d_m2)),
		"dx_min_m": float(np.min(dx_by_row_m)),
		"dx_max_m": float(np.max(dx_by_row_m)),
		"dx_mean_m": float(np.mean(dx_by_row_m)),
	}


def load_fire_geometry(
	data_dir: Path,
	config: Mapping[str, Any],
	geom_path: Path | None = None,
	terrain_path: Path | None = None,
	expected_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
	"""Load geometry, compute area map, and optionally validate against terrain and frame shape."""

	geometry_config = resolve_geometry_config(config)
	selected_geom_path = geom_path.resolve() if geom_path is not None else find_geom_file(data_dir)
	print(f"Selected geom file: {selected_geom_path}")
	selected_terrain_path = terrain_path.resolve() if terrain_path is not None else find_terrain_file(data_dir)
	geom_info = parse_geom_file(selected_geom_path)
	area_info = compute_cell_area_from_geom(geom_info, config)
	area_2d_m2 = np.asarray(area_info["area_2d_m2"], dtype=np.float32)

	if expected_shape is not None:
		if tuple(area_2d_m2.shape) != tuple(expected_shape):
			if geometry_config["allow_area_transpose_if_needed"] and tuple(area_2d_m2.T.shape) == tuple(expected_shape):
				print(
					f"WARNING: area map shape {tuple(area_2d_m2.shape)} did not match frame shape {tuple(expected_shape)}; transposing."
				)
				area_2d_m2 = np.ascontiguousarray(area_2d_m2.T, dtype=np.float32)
			else:
				raise ValueError(
					f"Area map shape from .geom does not match frame spatial shape. "
					f"area={tuple(area_2d_m2.shape)} frame={tuple(expected_shape)} geom={selected_geom_path}"
				)

	if geometry_config["validate_against_terrain_header"] and selected_terrain_path is not None:
		terrain_info = parse_terrain_header(selected_terrain_path)
		if (
			int(terrain_info["nx"]) != int(geom_info["nx"])
			or int(terrain_info["ny"]) != int(geom_info["ny"])
			or int(terrain_info["nz"]) != int(geom_info["nz"])
		):
			raise ValueError(
				"Terrain and geom dimensions do not match: "
				f"geom={(geom_info['nx'], geom_info['ny'], geom_info['nz'])} "
				f"terrain={(terrain_info['nx'], terrain_info['ny'], terrain_info['nz'])} "
				f"for {selected_geom_path}"
			)
		center_row = int(geom_info["ny"]) // 2
		center_dx_m = float(np.asarray(area_info["dx_by_row_m"], dtype=np.float32)[center_row])
		print(
			"Terrain header reference | "
			f"terrain={selected_terrain_path} dx_header_m={terrain_info['dx_header_m']:.6f} "
			f"dy_header_m={terrain_info['dy_header_m']:.6f} geom_center_dx_m={center_dx_m:.6f} "
			f"geom_dy_m={area_info['dy_m']:.6f}"
		)
	else:
		terrain_info = None
		if geometry_config["validate_against_terrain_header"]:
			print(f"WARNING: no terrain file found for geometry validation in {data_dir}")

	return {
		"geom_path": selected_geom_path,
		"terrain_path": selected_terrain_path,
		"terrain_header": terrain_info,
		"nx": int(geom_info["nx"]),
		"ny": int(geom_info["ny"]),
		"nz": int(geom_info["nz"]),
		"lons": np.asarray(geom_info["lons"], dtype=np.float32),
		"lats": np.asarray(geom_info["lats"], dtype=np.float32),
		"cos_lats_from_file": np.asarray(geom_info["cos_lats_from_file"], dtype=np.float32),
		**area_info,
		"area_2d_m2": np.asarray(area_2d_m2, dtype=np.float32),
	}
