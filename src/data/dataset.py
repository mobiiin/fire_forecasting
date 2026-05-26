"""PyTorch dataset and DataLoader helpers for sequence-to-map wildfire forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import numpy as np

try:
	import torch  # type: ignore[import-not-found]
	from torch.utils.data import DataLoader, Dataset  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None
	DataLoader = None

	class Dataset:  # type: ignore[too-many-ancestors]
		"""Fallback base class used only when PyTorch is unavailable."""

		pass

from src.data.preprocessing import load_normalization_stats, normalize_channel_map, normalize_tensor
from src.data.splits import chronological_split_indices


def _extract_numeric_suffix(name: str) -> int | None:
	"""Extract a trailing numeric suffix from a filename stem if present."""

	digits = []
	for character in reversed(name):
		if character.isdigit():
			digits.append(character)
		else:
			break
	if not digits:
		return None
	return int("".join(reversed(digits)))


def _sort_chronologically(file_paths: Sequence[Path]) -> list[Path]:
	"""Sort by trailing numeric suffix when available, otherwise lexicographically."""

	numeric_suffixes = [_extract_numeric_suffix(path.stem) for path in file_paths]
	if all(value is not None for value in numeric_suffixes):
		return [path for _, path in sorted(zip(numeric_suffixes, file_paths), key=lambda item: item[0])]
	return sorted(file_paths, key=lambda path: path.name)


def _resolve_path(base_path: Path | None, configured_path: str | Path) -> Path:
	"""Resolve a configured path relative to a config file when available."""

	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	if base_path is None:
		return path.resolve()
	return (base_path.parent / path).resolve()


def _as_path_list(file_paths: Iterable[str | Path]) -> list[Path]:
	"""Convert an arbitrary iterable of paths into a concrete list of Path objects."""

	return [Path(path) for path in file_paths]


class FireSequenceDataset(Dataset):
	"""Dataset for forecasting a target map from a sequence of preceding maps."""

	def __init__(
		self,
		file_paths: Iterable[str | Path],
		sample_indices: Sequence[int] | None,
		input_sequence_length: int,
		prediction_horizon: int,
		target_channel: int,
		input_channel_count: int | None = None,
		task_type: str = "regression",
		fire_threshold: float = 0.5,
		use_patches: bool = False,
		patch_size: int = 64,
		active_patch_probability: float = 0.7,
		active_threshold: float = 0.0,
		normalization_stats: Mapping[str, np.ndarray] | str | Path | None = None,
		normalize_target: bool = False,
		transform=None,
		target_transform=None,
		return_metadata: bool = False,
	) -> None:
		self.file_paths = _sort_chronologically(_as_path_list(file_paths))
		self.input_sequence_length = int(input_sequence_length)
		self.prediction_horizon = int(prediction_horizon)
		self.target_channel = int(target_channel)
		self.input_channel_count = None if input_channel_count is None else int(input_channel_count)
		self.task_type = str(task_type).lower()
		self.fire_threshold = float(fire_threshold)
		self.use_patches = bool(use_patches)
		self.patch_size = int(patch_size)
		self.active_patch_probability = float(active_patch_probability)
		self.active_threshold = float(active_threshold)
		self.transform = transform
		self.target_transform = target_transform
		self.return_metadata = bool(return_metadata)
		self.normalize_target = bool(normalize_target)

		if self.input_sequence_length <= 0:
			raise ValueError(
				f"input_sequence_length must be positive, got {self.input_sequence_length}."
			)
		if self.prediction_horizon < 0:
			raise ValueError(f"prediction_horizon must be non-negative, got {self.prediction_horizon}.")

		if not self.file_paths:
			raise ValueError("FireSequenceDataset requires at least one file path.")

		missing_files = [str(path) for path in self.file_paths if not path.exists()]
		if missing_files:
			raise FileNotFoundError(
				"The following dataset files do not exist:\n" + "\n".join(f"  {path}" for path in missing_files)
			)

		first_tensor = self._load_tensor(self.file_paths[0])
		if first_tensor.ndim != 3:
			raise ValueError(
				f"Expected dataset files to contain 3D tensors, got shape {first_tensor.shape} in {self.file_paths[0]}."
			)

		self.expected_height, self.expected_width, self.num_channels = first_tensor.shape
		if self.input_channel_count is None:
			self.input_channel_count = self.num_channels
		if self.input_channel_count <= 0 or self.input_channel_count > self.num_channels:
			raise ValueError(
				f"input_channel_count must be in [1, {self.num_channels}], got {self.input_channel_count}."
			)
		if self.target_channel < 0 or self.target_channel >= self.num_channels:
			raise ValueError(
				f"target_channel must be in [0, {self.num_channels - 1}], got {self.target_channel}."
			)
		if self.task_type not in {"regression", "segmentation"}:
			raise ValueError(f"task_type must be 'regression' or 'segmentation', got {self.task_type!r}.")
		if not 0.0 <= self.active_patch_probability <= 1.0:
			raise ValueError(
				"active_patch_probability must be in [0, 1], "
				f"got {self.active_patch_probability}."
			)
		if self.patch_size <= 0:
			raise ValueError(f"patch_size must be positive, got {self.patch_size}.")
		if self.use_patches and (
			self.patch_size > self.expected_height or self.patch_size > self.expected_width
		):
			raise ValueError(
				"patch_size must be <= both spatial dimensions. "
				f"Got patch_size={self.patch_size}, H={self.expected_height}, W={self.expected_width}."
			)

		max_valid_start = len(self.file_paths) - self.input_sequence_length - self.prediction_horizon
		if max_valid_start < 0:
			raise ValueError(
				"Not enough files to form even one sample. "
				f"Need at least input_sequence_length + prediction_horizon = "
				f"{self.input_sequence_length + self.prediction_horizon}, got {len(self.file_paths)}."
			)

		if sample_indices is None:
			self.sample_indices = list(range(max_valid_start + 1))
		else:
			self.sample_indices = [int(index) for index in sample_indices]

		invalid_indices = [index for index in self.sample_indices if index < 0 or index > max_valid_start]
		if invalid_indices:
			raise ValueError(
				"sample_indices contain invalid sample start positions. "
				f"Valid range is [0, {max_valid_start}], got {invalid_indices[:10]}."
			)

		self.normalization_stats = self._coerce_normalization_stats(normalization_stats)
		self.target_mean, self.target_std = self._resolve_target_normalization_stats()

	def _coerce_normalization_stats(
		self,
		normalization_stats: Mapping[str, np.ndarray] | str | Path | None,
	) -> dict[str, np.ndarray] | None:
		"""Normalize the normalization-statistics input into a usable dictionary."""

		if normalization_stats is None:
			return None
		if isinstance(normalization_stats, (str, Path)):
			stats = load_normalization_stats(normalization_stats)
		else:
			stats = dict(normalization_stats)

		required_keys = {"mean", "std", "min", "max"}
		missing = required_keys.difference(stats)
		if missing:
			raise KeyError(
				f"Normalization stats are missing required key(s): {', '.join(sorted(missing))}"
			)

		normalized_stats = {key: np.asarray(stats[key]) for key in required_keys}
		for optional_key in ("target_mean", "target_std", "target_min", "target_max"):
			if optional_key in stats:
				normalized_stats[optional_key] = np.asarray(stats[optional_key])
		return normalized_stats

	def _resolve_target_normalization_stats(self) -> tuple[float | None, float | None]:
		"""Resolve scalar stats for target normalization when enabled."""

		if self.normalization_stats is None or not self.normalize_target or self.task_type != "regression":
			return None, None

		stats_mean = np.asarray(self.normalization_stats["mean"])
		stats_std = np.asarray(self.normalization_stats["std"])
		if self.target_channel < stats_mean.shape[0] and self.target_channel < stats_std.shape[0]:
			return float(stats_mean[self.target_channel]), float(stats_std[self.target_channel])

		if "target_mean" in self.normalization_stats and "target_std" in self.normalization_stats:
			return (
				float(np.asarray(self.normalization_stats["target_mean"])),
				float(np.asarray(self.normalization_stats["target_std"])),
			)

		raise ValueError(
			"Target normalization was requested, but normalization stats do not include "
			f"target channel {self.target_channel}. Recompute normalization stats with target stats enabled."
		)

	def _load_tensor(self, file_path: Path) -> np.ndarray:
		"""Load a single tensor from disk with validation-friendly settings."""

		return np.load(file_path, mmap_mode="r", allow_pickle=False)

	def _validate_tensor_shape(self, tensor: np.ndarray, file_path: Path) -> None:
		"""Ensure a loaded tensor matches the expected spatial and channel dimensions."""

		if tensor.ndim != 3:
			raise ValueError(f"Expected a 3D tensor in {file_path}, got shape {tensor.shape}.")
		if tensor.shape != (self.expected_height, self.expected_width, self.num_channels):
			raise ValueError(
				f"Inconsistent tensor shape in {file_path}. "
				f"Expected {(self.expected_height, self.expected_width, self.num_channels)}, got {tensor.shape}."
			)

	def _sample_patch_origin(self, target_map_for_sampling: np.ndarray) -> tuple[int, int]:
		"""Choose the top-left patch origin, preferring active fire areas when requested."""

		height, width = target_map_for_sampling.shape
		max_top = height - self.patch_size
		max_left = width - self.patch_size

		use_active_patch = np.random.random() < self.active_patch_probability
		if use_active_patch:
			active_pixels = np.argwhere(target_map_for_sampling > self.active_threshold)
			if active_pixels.size > 0:
				center_y, center_x = active_pixels[np.random.randint(active_pixels.shape[0])]
				top = int(center_y) - self.patch_size // 2
				left = int(center_x) - self.patch_size // 2
				top = max(0, min(top, max_top))
				left = max(0, min(left, max_left))
				return top, left

		top = int(np.random.randint(0, max_top + 1)) if max_top > 0 else 0
		left = int(np.random.randint(0, max_left + 1)) if max_left > 0 else 0
		return top, left

	def __len__(self) -> int:
		return len(self.sample_indices)

	def __getitem__(self, index: int):
		if torch is None:
			raise ImportError("PyTorch is required to index FireSequenceDataset.")

		sample_start = self.sample_indices[index]
		input_file_paths = self.file_paths[sample_start : sample_start + self.input_sequence_length]
		target_index = sample_start + self.input_sequence_length - 1 + self.prediction_horizon
		target_file_path = self.file_paths[target_index]

		input_frames = []
		for file_path in input_file_paths:
			tensor = self._load_tensor(file_path)
			self._validate_tensor_shape(tensor, file_path)
			input_frames.append(np.asarray(tensor[:, :, : self.input_channel_count], dtype=np.float32))

		target_tensor = self._load_tensor(target_file_path)
		self._validate_tensor_shape(target_tensor, target_file_path)
		raw_target_array = np.asarray(target_tensor[:, :, self.target_channel], dtype=np.float32)
		target_array = raw_target_array.copy()
		if self.task_type == "segmentation":
			target_array = (target_array > self.fire_threshold).astype(np.float32, copy=False)

		stacked_inputs = np.stack(input_frames, axis=0)
		patch_top = None
		patch_left = None
		if self.use_patches:
			patch_top, patch_left = self._sample_patch_origin(raw_target_array)
			patch_bottom = patch_top + self.patch_size
			patch_right = patch_left + self.patch_size
			stacked_inputs = stacked_inputs[:, patch_top:patch_bottom, patch_left:patch_right, :]
			target_array = target_array[patch_top:patch_bottom, patch_left:patch_right]
			if stacked_inputs.shape[1] != self.patch_size or stacked_inputs.shape[2] != self.patch_size:
				raise ValueError("Input patch extraction produced an unexpected shape.")
			if target_array.shape != (self.patch_size, self.patch_size):
				raise ValueError("Target patch extraction produced an unexpected shape.")
		if self.normalization_stats is not None:
			stats_mean = np.asarray(self.normalization_stats["mean"])
			stats_std = np.asarray(self.normalization_stats["std"])
			if stats_mean.shape[0] != self.input_channel_count:
				if stats_mean.shape[0] < self.input_channel_count:
					raise ValueError(
						"Normalization stats have fewer channels than the dataset input slice. "
						f"Got {stats_mean.shape[0]} stats channels and {self.input_channel_count} input channels."
					)
				stats_mean = stats_mean[: self.input_channel_count]
				stats_std = stats_std[: self.input_channel_count]
			stacked_inputs = normalize_tensor(
				stacked_inputs,
				stats_mean,
				stats_std,
			).astype(np.float32, copy=False)
		if self.target_mean is not None and self.target_std is not None:
			target_array = normalize_channel_map(
				target_array,
				self.target_mean,
				self.target_std,
			).astype(np.float32, copy=False)

		stacked_inputs = np.transpose(stacked_inputs, (0, 3, 1, 2))
		stacked_inputs = np.ascontiguousarray(stacked_inputs, dtype=np.float32)
		target_array = np.expand_dims(target_array, axis=0)
		target_array = np.ascontiguousarray(target_array, dtype=np.float32)

		x_tensor = torch.from_numpy(stacked_inputs).to(torch.float32)
		y_tensor = torch.from_numpy(target_array).to(torch.float32)

		if self.transform is not None:
			x_tensor = self.transform(x_tensor)
		if self.target_transform is not None:
			y_tensor = self.target_transform(y_tensor)

		if self.return_metadata:
			metadata = {
				"sample_index": sample_start,
				"target_file_path": str(target_file_path),
			}
			if self.use_patches:
				metadata["patch_top"] = int(patch_top)
				metadata["patch_left"] = int(patch_left)
				metadata["patch_size"] = int(self.patch_size)
			return x_tensor, y_tensor, metadata

		return x_tensor, y_tensor


def create_dataloaders(config):
	"""Build train/validation/test DataLoaders from a configuration dictionary.

	Relative paths are resolved against ``config["config_path"]`` or
	``config["_config_path"]`` when present. If neither is available, paths are
	resolved relative to the current working directory.
	"""

	if torch is None or DataLoader is None:
		raise ImportError("PyTorch is required to build DataLoaders for wildfire forecasting.")

	if "data_dir" not in config:
		raise KeyError("Config is missing required key 'data_dir'.")
	if "file_pattern" not in config:
		raise KeyError("Config is missing required key 'file_pattern'.")
	if "input_sequence_length" not in config:
		raise KeyError("Config is missing required key 'input_sequence_length'.")
	if "prediction_horizon" not in config:
		raise KeyError("Config is missing required key 'prediction_horizon'.")
	if "target_channel" not in config:
		raise KeyError("Config is missing required key 'target_channel'.")

	config_path_value = config.get("config_path", config.get("_config_path"))
	config_path = Path(config_path_value).expanduser().resolve() if config_path_value else None
	data_dir = _resolve_path(config_path, config["data_dir"])
	file_pattern = str(config["file_pattern"])

	if not data_dir.exists():
		raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

	files = _sort_chronologically(list(data_dir.glob(file_pattern)))
	if not files:
		raise FileNotFoundError(f"No files found in '{data_dir}' using pattern '{file_pattern}'.")

	input_sequence_length = int(config["input_sequence_length"])
	prediction_horizon = int(config["prediction_horizon"])
	target_channel = int(config["target_channel"])
	input_channel_count = int(config.get("input_channel_count", config.get("model", {}).get("input_channels", 0)))
	if input_channel_count <= 0:
		raise KeyError("Config must define a positive input_channel_count or model.input_channels.")

	split_indices = chronological_split_indices(
		num_timesteps=len(files),
		input_sequence_length=input_sequence_length,
		prediction_horizon=prediction_horizon,
		train_fraction=float(config.get("train_fraction", 0.7)),
		val_fraction=float(config.get("val_fraction", 0.15)),
		test_fraction=float(config.get("test_fraction", 0.15)),
	)

	normalization_stats = None
	normalization_config = config.get("normalization", {})
	normalization_path = normalization_config.get("path")
	normalize_target = bool(normalization_config.get("normalize_target", False))
	if normalization_path:
		resolved_normalization_path = _resolve_path(config_path, normalization_path)
		if resolved_normalization_path.exists():
			normalization_stats = resolved_normalization_path

	dataset_kwargs = {
		"file_paths": files,
		"input_sequence_length": input_sequence_length,
		"prediction_horizon": prediction_horizon,
		"target_channel": target_channel,
		"input_channel_count": input_channel_count,
		"task_type": str(config.get("task_type", config.get("training", {}).get("task_type", "regression"))),
		"fire_threshold": float(config.get("fire_threshold", config.get("training", {}).get("fire_threshold", 0.5))),
		"patch_size": int(config.get("patch_size", 64)),
		"active_patch_probability": float(config.get("active_patch_probability", 0.7)),
		"active_threshold": float(config.get("active_threshold", config.get("fire_threshold", 0.5))),
		"normalization_stats": normalization_stats,
		"normalize_target": normalize_target,
	}
	use_train_patches = bool(config.get("use_patches", False))
	use_eval_patches = bool(config.get("use_patches_for_eval", False))

	train_dataset = FireSequenceDataset(
		sample_indices=split_indices["train"],
		use_patches=use_train_patches,
		**dataset_kwargs,
	)
	val_dataset = FireSequenceDataset(
		sample_indices=split_indices["val"],
		use_patches=use_eval_patches,
		**dataset_kwargs,
	)
	test_dataset = FireSequenceDataset(
		sample_indices=split_indices["test"],
		use_patches=use_eval_patches,
		**dataset_kwargs,
	)

	batch_size = int(config.get("batch_size", 4))
	num_workers = int(config.get("num_workers", 0))

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		drop_last=False,
	)

	return train_loader, val_loader, test_loader


if __name__ == "__main__":
	from pathlib import Path

	from src.config import load_config

	if torch is None:
		print("smoke test skipped: PyTorch is not installed in this environment")
		raise SystemExit(0)

	project_root = Path(__file__).resolve().parents[2]
	config_path = project_root / "configs" / "default.yaml"
	config = load_config(config_path)
	config["config_path"] = str(config_path)

	train_loader, val_loader, test_loader = create_dataloaders(config)

	batch = next(iter(train_loader))
	x_batch, y_batch = batch[:2]
	assert x_batch.ndim == 5, f"Expected X batch to have 5 dimensions, got {tuple(x_batch.shape)}."
	assert y_batch.ndim == 4, f"Expected y batch to have 4 dimensions, got {tuple(y_batch.shape)}."
	expected_spatial = int(config.get("patch_size", 64)) if bool(config.get("use_patches", False)) else 144
	assert x_batch.shape[1:] == (
		int(config["input_sequence_length"]),
		int(config.get("input_channel_count", config["model"]["input_channels"])),
		expected_spatial,
		expected_spatial,
	), f"Unexpected X batch shape: {tuple(x_batch.shape)}"
	assert y_batch.shape[1:] == (1, expected_spatial, expected_spatial), f"Unexpected y batch shape: {tuple(y_batch.shape)}"
	print(f"smoke test passed: X={tuple(x_batch.shape)}, y={tuple(y_batch.shape)}")
