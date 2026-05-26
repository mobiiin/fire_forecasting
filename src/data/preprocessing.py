"""Preprocessing utilities for wildfire tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def compute_channel_stats(
	file_paths: Iterable[str | Path],
	sample_indices: Sequence[int] | None = None,
	channel_indices: Sequence[int] | slice | None = None,
	eps: float = 1e-6,
) -> dict[str, np.ndarray]:
	"""Compute per-channel statistics with a numerically stable streaming update.

	Each file is expected to contain a tensor shaped ``(H, W, C)``. Statistics are
	accumulated over all pixels from all selected files without loading the full
	dataset into memory.
	"""

	resolved_paths = [Path(path) for path in file_paths]
	if sample_indices is not None:
		resolved_paths = [resolved_paths[index] for index in sample_indices]

	if not resolved_paths:
		raise ValueError("No files were provided for normalization statistics.")

	count = 0
	mean = None
	m2 = None
	channel_min = None
	channel_max = None

	for file_path in resolved_paths:
		array = np.load(file_path, allow_pickle=False)
		if array.ndim != 3:
			raise ValueError(f"Expected a 3D tensor in {file_path}, got shape {array.shape}.")
		if channel_indices is not None:
			array = array[:, :, channel_indices]
			if array.ndim != 3:
				raise ValueError(
					"channel_indices must select at least one channel. "
					f"Got resulting shape {array.shape} for {file_path}."
				)

		if not np.issubdtype(array.dtype, np.floating):
			array = array.astype(np.float64, copy=False)
		else:
			array = array.astype(np.float64, copy=False)

		flat = array.reshape(-1, array.shape[-1])
		file_count = flat.shape[0]

		file_mean = flat.mean(axis=0)
		file_min = flat.min(axis=0)
		file_max = flat.max(axis=0)
		centered = flat - file_mean
		file_m2 = np.sum(centered * centered, axis=0)

		if mean is None:
			mean = file_mean
			m2 = file_m2
			channel_min = file_min
			channel_max = file_max
			count = file_count
			continue

		delta = file_mean - mean
		total_count = count + file_count
		mean = mean + delta * (file_count / total_count)
		m2 = m2 + file_m2 + (delta * delta) * (count * file_count / total_count)
		channel_min = np.minimum(channel_min, file_min)
		channel_max = np.maximum(channel_max, file_max)
		count = total_count

	assert mean is not None
	assert m2 is not None
	assert channel_min is not None
	assert channel_max is not None

	variance = m2 / max(count, 1)
	std = np.sqrt(np.maximum(variance, 0.0))
	std = np.maximum(std, eps)

	return {
		"mean": mean,
		"std": std,
		"min": channel_min,
		"max": channel_max,
	}


def normalize_tensor(
	x: np.ndarray,
	mean: np.ndarray,
	std: np.ndarray,
) -> np.ndarray:
	"""Normalize a tensor shaped ``(H, W, C)`` or ``(T, H, W, C)`` channel-wise."""

	array = np.asarray(x)
	mean_array = np.asarray(mean)
	std_array = np.asarray(std)

	if array.ndim not in (3, 4):
		raise ValueError(f"normalize_tensor expects a 3D or 4D tensor, got shape {array.shape}.")
	if array.shape[-1] != mean_array.shape[0] or mean_array.shape != std_array.shape:
		raise ValueError(
			"Mean/std shapes must match the channel dimension of the input tensor. "
			f"Got x.shape={array.shape}, mean.shape={mean_array.shape}, std.shape={std_array.shape}."
		)

	safe_std = np.maximum(std_array, 1e-6)
	return (array - mean_array) / safe_std


def normalize_channel_map(
	x: np.ndarray,
	mean: float | np.ndarray,
	std: float | np.ndarray,
) -> np.ndarray:
	"""Normalize a single 2D channel map with scalar statistics."""

	array = np.asarray(x, dtype=np.float32)
	mean_value = float(np.asarray(mean, dtype=np.float32))
	std_value = max(float(np.asarray(std, dtype=np.float32)), 1e-6)
	return (array - mean_value) / std_value


def inverse_normalize_channel_map(
	x: np.ndarray,
	mean: float | np.ndarray,
	std: float | np.ndarray,
) -> np.ndarray:
	"""Undo scalar normalization for a single 2D channel map."""

	array = np.asarray(x, dtype=np.float32)
	mean_value = float(np.asarray(mean, dtype=np.float32))
	std_value = max(float(np.asarray(std, dtype=np.float32)), 1e-6)
	return array * std_value + mean_value


def load_normalization_stats(path: str | Path) -> dict[str, np.ndarray]:
	"""Load normalization statistics from a saved ``.npz`` archive."""

	archive_path = Path(path).expanduser().resolve()
	if not archive_path.exists():
		raise FileNotFoundError(f"Normalization statistics file not found: {archive_path}")

	with np.load(archive_path, allow_pickle=False) as data:
		required_keys = {"mean", "std", "min", "max"}
		missing = required_keys.difference(data.files)
		if missing:
			raise KeyError(
				f"Normalization archive is missing required key(s): {', '.join(sorted(missing))}"
			)
		stats = {key: data[key] for key in required_keys}
		for optional_key in ("target_mean", "target_std", "target_min", "target_max"):
			if optional_key in data.files:
				stats[optional_key] = data[optional_key]
		return stats
