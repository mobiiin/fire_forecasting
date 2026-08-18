"""Parsing and feature extraction for CAWFE plain-text terrain grids."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def find_terrain_file(fire_source_dir: str | Path) -> Path | None:
    paths = sorted(Path(fire_source_dir).glob("*.terrain"))
    if len(paths) > 1:
        raise ValueError(f"Multiple .terrain files found in {fire_source_dir}: {[p.name for p in paths]}")
    return paths[0] if paths else None


def _header_from_line(line: str) -> dict[str, Any] | None:
    values = _NUMBER.findall(line)
    if len(values) < 5:
        return None
    numbers = [float(value) for value in values[:5]]
    if not all(float(value).is_integer() for value in numbers[:3]):
        return None
    return {"nx": int(numbers[0]), "ny": int(numbers[1]), "nz": int(numbers[2]), "dx": numbers[3], "dy": numbers[4]}


def parse_terrain_file(path: str | Path, expected_shape: tuple[int, int] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Terrain file not found: {path}")
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise ValueError(f"Terrain file is empty: {path}")
    header = _header_from_line(lines[0])
    # Match the ASC/grid loader convention used by bigtensor.py: the first
    # line is metadata, and only the remaining numeric grid values are loaded.
    try:
        values = np.asarray(np.loadtxt(path, dtype=np.float64, skiprows=1), dtype=np.float64).reshape(-1)
    except ValueError as exc:
        raise ValueError(f"Could not load numeric terrain values after the header in {path}: {exc}") from exc
    if header is not None:
        nx, ny = header["nx"], header["ny"]
        needed = nx * ny
        if values.size < needed:
            raise ValueError(f"Terrain file {path} contains {values.size} values, expected at least {needed} for {nx}x{ny}.")
        values = values[:needed]
        # Terrain files report nx, ny in the header, but the remaining text is
        # laid out as ny rows by nx columns. Reconstruct that source grid first,
        # then orient it to the processed frame shape below. This changes the
        # value ordering, not just the displayed image orientation.
        source_grid = values.reshape((ny, nx), order="C")
        if expected_shape is None:
            candidates = [(source_grid.T, True, "ny,nx->nx,ny", 0, 1)]
        elif tuple(expected_shape) == (nx, ny):
            candidates = [(source_grid.T, True, "ny,nx->nx,ny", 0, 1)]
        elif tuple(expected_shape) == (ny, nx):
            candidates = [(source_grid, False, "ny,nx", 1, 0)]
        else:
            candidates = []
    elif expected_shape is not None and values.size == int(expected_shape[0]) * int(expected_shape[1]):
        candidates = [(values.reshape(expected_shape, order="C"), False, "expected_shape", 1, 0)]
        nx, ny = int(expected_shape[1]), int(expected_shape[0])
        header = {"nx": nx, "ny": ny, "nz": None, "dx": None, "dy": None}
    else:
        side = int(round(values.size ** 0.5))
        if side * side != values.size:
            raise ValueError(f"Cannot infer terrain H,W from {values.size} values in {path}; provide a header or expected frame shape.")
        candidates = [(values.reshape((side, side), order="C"), False, "square_inferred", 1, 0)]
        nx, ny = side, side
        header = {"nx": nx, "ny": ny, "nz": None, "dx": None, "dy": None}
    transposed = False
    reconstruction = "direct"
    x_axis = 1
    y_axis = 0
    if expected_shape is not None:
        matches = [candidate for candidate in candidates if tuple(candidate[0].shape) == tuple(expected_shape)]
        if not matches:
            candidate_shapes = [candidate[0].shape for candidate in candidates]
            raise ValueError(f"Terrain shape does not match frame shape: terrain candidates={candidate_shapes} frame={expected_shape} path={path}")
        height, transposed, reconstruction, x_axis, y_axis = matches[0]
    else:
        height, transposed, reconstruction, x_axis, y_axis = candidates[0]
    height = np.asarray(height, dtype=np.float32)
    metadata = {"terrain_file": str(path), "header": header, "source_grid_shape": [int(ny), int(nx)], "height_shape_saved": list(height.shape), "transposed": bool(transposed), "reconstruction": reconstruction, "x_axis": int(x_axis), "y_axis": int(y_axis), "dx": header.get("dx"), "dy": header.get("dy"), "units": "meters", "reshape_order": "C", "layout": "frame_spatial"}
    return height, metadata


def _stats_for_channel(array: np.ndarray) -> dict[str, float]:
    values = np.asarray(array, dtype=np.float32)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def validate_terrain_features(features: np.ndarray, expected_shape: tuple[int, int] | None = None, context: str = "terrain_features") -> None:
    array = np.asarray(features)
    if array.dtype != np.float32:
        raise ValueError(f"{context} must be float32 before saving, got dtype={array.dtype}.")
    if array.ndim != 3 or array.shape[0] != 4:
        raise ValueError(f"{context} must have shape (4,H,W), got {array.shape}.")
    if expected_shape is not None and tuple(array.shape[1:]) != tuple(expected_shape):
        raise ValueError(f"{context} spatial shape {array.shape[1:]} does not match expected {expected_shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{context} contains NaN/Inf values.")
    ranges = ((0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    names = ("relative_elevation", "slope_magnitude", "slope_x", "slope_y")
    eps = 1e-6
    for index, (low, high) in enumerate(ranges):
        channel = array[index]
        cmin = float(channel.min())
        cmax = float(channel.max())
        if cmin < low - eps or cmax > high + eps:
            raise ValueError(
                f"{context} channel {index} ({names[index]}) is outside expected range "
                f"[{low}, {high}]: min={cmin}, max={cmax}."
            )


def compute_terrain_features(terrain_height: np.ndarray, dx: float | None = None, dy: float | None = None, normalization_config: Mapping[str, Any] | None = None, x_axis: int = 1, y_axis: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    config = dict(normalization_config or {})
    height = np.asarray(terrain_height, dtype=np.float32)
    if height.ndim != 2 or not np.isfinite(height).all():
        raise ValueError(f"terrain_height must be finite HxW, got {height.shape}")
    eps = float(config.get("eps", 1e-6))

    rel_elev_p1, rel_elev_p99 = np.percentile(height, [1, 99])
    relative_elevation = np.clip((height - rel_elev_p1) / (rel_elev_p99 - rel_elev_p1 + eps), 0.0, 1.0)

    dx_value = float(dx) if dx not in (None, 0) else 1.0
    dy_value = float(dy) if dy not in (None, 0) else 1.0
    x_axis = int(x_axis)
    y_axis = int(y_axis)
    if sorted((x_axis, y_axis)) != [0, 1]:
        raise ValueError(f"x_axis/y_axis must be 0 and 1 for terrain features, got x_axis={x_axis}, y_axis={y_axis}")
    axis_spacings = [1.0, 1.0]
    axis_spacings[x_axis] = dx_value
    axis_spacings[y_axis] = dy_value
    gradients = np.gradient(height, *axis_spacings)
    slope_x = gradients[x_axis]
    slope_y = gradients[y_axis]

    slope_mag = np.sqrt(slope_x * slope_x + slope_y * slope_y)
    slope_mag_log = np.log1p(slope_mag)
    slope_mag_p1, slope_mag_p99 = np.percentile(slope_mag_log, [1, 99])
    slope_mag_norm = np.clip((slope_mag_log - slope_mag_p1) / (slope_mag_p99 - slope_mag_p1 + eps), 0.0, 1.0)

    slope_x_abs_p99 = max(float(np.percentile(np.abs(slope_x), 99)), eps)
    slope_y_abs_p99 = max(float(np.percentile(np.abs(slope_y), 99)), eps)
    slope_x_norm = np.clip(slope_x / (slope_x_abs_p99 + eps), -1.0, 1.0)
    slope_y_norm = np.clip(slope_y / (slope_y_abs_p99 + eps), -1.0, 1.0)

    features = np.stack([relative_elevation, slope_mag_norm, slope_x_norm, slope_y_norm]).astype(np.float32)
    validate_terrain_features(features, expected_shape=height.shape)

    channel_names = ("relative_elevation", "slope_magnitude", "slope_x", "slope_y")
    final_channel_stats = {name: _stats_for_channel(features[index]) for index, name in enumerate(channel_names)}
    metadata = {
        "feature_shape": list(features.shape),
        "feature_channels": [{"index": i, "name": name} for i, name in enumerate(channel_names)],
        "normalization": {
            "relative_elevation": "per_fire_p1_p99_minmax_clip_0_1",
            "slope_magnitude": "raw_gradient_log1p_per_fire_p1_p99_minmax_clip_0_1",
            "slope_x": "raw_gradient_per_fire_p99_abs_clip_neg1_1",
            "slope_y": "raw_gradient_per_fire_p99_abs_clip_neg1_1",
        },
        "rel_elev_p1": float(rel_elev_p1),
        "rel_elev_p99": float(rel_elev_p99),
        "rel_elev_min": final_channel_stats["relative_elevation"]["min"],
        "rel_elev_max": final_channel_stats["relative_elevation"]["max"],
        "slope_mag_p1": float(slope_mag_p1),
        "slope_mag_p99": float(slope_mag_p99),
        "slope_x_abs_p99": float(slope_x_abs_p99),
        "slope_y_abs_p99": float(slope_y_abs_p99),
        "final_channel_stats": final_channel_stats,
        "stats": {
            "height_min": float(height.min()),
            "height_mean": float(height.mean()),
            "height_std": float(height.std()),
            "height_max": float(height.max()),
            "rel_elev_p1": float(rel_elev_p1),
            "rel_elev_p99": float(rel_elev_p99),
            "rel_elev_min": final_channel_stats["relative_elevation"]["min"],
            "rel_elev_max": final_channel_stats["relative_elevation"]["max"],
            "slope_mag_p1": float(slope_mag_p1),
            "slope_mag_p99": float(slope_mag_p99),
            "slope_x_abs_p99": float(slope_x_abs_p99),
            "slope_y_abs_p99": float(slope_y_abs_p99),
        },
        "dx": dx,
        "dy": dy,
        "x_axis": x_axis,
        "y_axis": y_axis,
    }
    return features, metadata
