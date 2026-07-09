"""Dataset for precomputed wildfire patch-cache shards."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
	import torch  # type: ignore[import-not-found]
	from torch.utils.data import Dataset  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific fallback
	torch = None

	class Dataset:  # type: ignore[too-many-ancestors]
		pass

from src.data.cache import get_patch_cache_dir, load_cache_manifest
from src.data.preprocessing import load_normalization_stats, normalize_tensor


def _get_section(config: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
	if not isinstance(config, Mapping):
		return {}
	section = config.get(name)
	return dict(section) if isinstance(section, Mapping) else {}


def _resolve_path(config: Mapping[str, Any], configured_path: str | Path) -> Path:
	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	config_path_value = config.get("config_path", config.get("_config_path"))
	if config_path_value:
		return (Path(config_path_value).expanduser().resolve().parent / path).resolve()
	return path.resolve()


def _coerce_normalization_stats(
	normalization_stats: Mapping[str, np.ndarray] | str | Path | None,
) -> dict[str, np.ndarray] | None:
	if normalization_stats is None:
		return None
	if isinstance(normalization_stats, (str, Path)):
		return load_normalization_stats(normalization_stats)
	return {str(key): np.asarray(value) for key, value in normalization_stats.items()}


def _read_metadata_jsonl(path: Path) -> list[dict[str, Any]]:
	if not path.exists():
		return []
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


class CachedPatchDataset(Dataset):
	"""Load precomputed patch shards shaped for ConvLSTM U-Net training."""

	def __init__(
		self,
		cache_dir: str | Path | None,
		split: str,
		config: Mapping[str, Any],
		normalization_stats: Mapping[str, np.ndarray] | str | Path | None = None,
		return_metadata: bool = False,
	) -> None:
		if torch is None:
			raise ImportError("PyTorch is required to use CachedPatchDataset.")

		self.config = dict(config)
		self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else get_patch_cache_dir(config)
		self.split = str(split).lower()
		self.return_metadata = bool(return_metadata)
		self.manifest = load_cache_manifest(self.cache_dir)
		self.cache_config = _get_section(config, "cache")
		self.shard_memory_cache_size = max(0, int(self.cache_config.get("shard_memory_cache_size", 2)))
		self.save_normalized_inputs = bool(
			self.manifest.get("save_normalized_inputs", self.cache_config.get("save_normalized_inputs", False))
		)
		self.normalization_stats = _coerce_normalization_stats(normalization_stats)
		self.task_type = str(config.get("task_type", self.manifest.get("task_type", "multitask"))).lower()
		self.normalize_target = False
		self.target_mean = None
		self.target_std = None

		shards_by_split = self.manifest.get("shards", {})
		if not isinstance(shards_by_split, Mapping) or self.split not in shards_by_split:
			raise ValueError(f"Patch-cache manifest does not contain split={self.split!r}.")
		shard_entries = shards_by_split[self.split]
		if not isinstance(shard_entries, list) or not shard_entries:
			raise ValueError(f"Patch-cache manifest contains no shards for split={self.split!r}.")

		self.shards: list[dict[str, Any]] = []
		self._offsets: list[int] = [0]
		for shard_index, entry in enumerate(shard_entries):
			if isinstance(entry, str):
				relative_path = entry
				num_samples = None
			elif isinstance(entry, Mapping):
				relative_path = entry.get("path")
				num_samples = entry.get("num_samples")
			else:
				raise TypeError(f"Unsupported shard entry type for split={self.split!r}: {type(entry)!r}")
			if relative_path in (None, "", "null"):
				raise ValueError(f"Shard entry is missing a path: {entry!r}")
			shard_path = Path(str(relative_path)).expanduser()
			if not shard_path.is_absolute():
				shard_path = self.cache_dir / shard_path
			shard_path = shard_path.resolve()
			if num_samples is None:
				x_shape, _ = self._read_shard_shapes(shard_path)
				num_samples = x_shape[0]
			num_samples = int(num_samples)
			if num_samples <= 0:
				raise ValueError(f"Shard has no samples: {shard_path}")
			self.shards.append(
				{
					"index": shard_index,
					"path": shard_path,
					"num_samples": num_samples,
				}
			)
			self._offsets.append(self._offsets[-1] + num_samples)
		self._length = int(self._offsets[-1])
		if self._length <= 0:
			raise ValueError(f"CachedPatchDataset split={self.split!r} is empty.")

		self.metadata = _read_metadata_jsonl(self.cache_dir / self.split / "metadata.jsonl")
		if self.metadata and len(self.metadata) != self._length:
			raise ValueError(
				f"Metadata length mismatch for split={self.split!r}: "
				f"metadata rows={len(self.metadata)}, cached samples={self._length}."
			)

		self._shard_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
		self.input_sequence_length = int(self.manifest.get("input_sequence_length", config.get("input_sequence_length", 0)))
		self.total_input_channels = int(self.manifest.get("input_channels", _get_section(config, "model").get("input_channels", 0)))
		self.input_channels_after_engineering = self.total_input_channels
		self.num_channels = self.total_input_channels
		self.base_input_channel_count = int(self.manifest.get("base_input_channel_count", config.get("input_channel_count", self.total_input_channels)))
		self.fuel_flux_engineered_channel_count = int(self.manifest.get("fuel_flux_engineered_channel_count", 0))
		self.atmospheric_engineered_channel_count = int(self.manifest.get("atmospheric_engineered_channel_count", 0))
		self.energy_history_channel_count = int(self.manifest.get("energy_history_channel_count", 0))
		self.engineered_channel_count = int(
			self.manifest.get(
				"engineered_channel_count",
				self.total_input_channels - self.base_input_channel_count,
			)
		)

	def __len__(self) -> int:
		return self._length

	def _read_shard_shapes(self, shard_path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
		if shard_path.suffix.lower() == ".npz":
			with np.load(shard_path, allow_pickle=False) as data:
				return tuple(int(value) for value in data["X"].shape), tuple(int(value) for value in data["y"].shape)
		if shard_path.suffix.lower() == ".pt":
			shard = torch.load(shard_path, map_location="cpu")
			return tuple(int(value) for value in shard["X"].shape), tuple(int(value) for value in shard["y"].shape)
		raise ValueError(f"Unsupported patch-cache shard format: {shard_path}")

	def _locate(self, index: int) -> tuple[int, int]:
		if index < 0:
			index += self._length
		if index < 0 or index >= self._length:
			raise IndexError(index)
		shard_index = bisect_right(self._offsets, index) - 1
		local_index = index - self._offsets[shard_index]
		return shard_index, local_index

	def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
		shard_info = self.shards[shard_index]
		shard_path = Path(shard_info["path"])
		cache_key = str(shard_path)
		if cache_key in self._shard_cache:
			self._shard_cache.move_to_end(cache_key)
			return self._shard_cache[cache_key]

		if shard_path.suffix.lower() == ".npz":
			with np.load(shard_path, allow_pickle=False) as data:
				shard = {
					"X": np.asarray(data["X"], dtype=np.float32),
					"y": np.asarray(data["y"], dtype=np.float32),
				}
		elif shard_path.suffix.lower() == ".pt":
			raw = torch.load(shard_path, map_location="cpu")
			shard = {
				"X": raw["X"].detach().cpu().numpy().astype(np.float32, copy=False) if torch.is_tensor(raw["X"]) else np.asarray(raw["X"], dtype=np.float32),
				"y": raw["y"].detach().cpu().numpy().astype(np.float32, copy=False) if torch.is_tensor(raw["y"]) else np.asarray(raw["y"], dtype=np.float32),
			}
		else:
			raise ValueError(f"Unsupported patch-cache shard format: {shard_path}")

		if self.shard_memory_cache_size > 0:
			self._shard_cache[cache_key] = shard
			self._shard_cache.move_to_end(cache_key)
			while len(self._shard_cache) > self.shard_memory_cache_size:
				self._shard_cache.popitem(last=False)
		return shard

	def _normalize_x(self, x_array: np.ndarray) -> np.ndarray:
		if self.save_normalized_inputs or self.normalization_stats is None:
			return x_array
		stats_mean = np.asarray(self.normalization_stats["mean"], dtype=np.float32)
		stats_std = np.asarray(self.normalization_stats["std"], dtype=np.float32)
		if stats_mean.shape[0] != x_array.shape[1] or stats_std.shape[0] != x_array.shape[1]:
			raise ValueError(
				"Normalization stats channel count does not match cached X. "
				f"Need {x_array.shape[1]}, got mean={stats_mean.shape[0]} std={stats_std.shape[0]}."
			)
		x_last = np.transpose(x_array, (0, 2, 3, 1))
		x_last = normalize_tensor(x_last, stats_mean, stats_std).astype(np.float32, copy=False)
		return np.ascontiguousarray(np.transpose(x_last, (0, 3, 1, 2)), dtype=np.float32)

	def __getitem__(self, index: int):
		shard_index, local_index = self._locate(int(index))
		shard = self._load_shard(shard_index)
		x_array = np.asarray(shard["X"][local_index], dtype=np.float32)
		y_array = np.asarray(shard["y"][local_index], dtype=np.float32)
		x_array = self._normalize_x(x_array)
		x_tensor = torch.from_numpy(np.ascontiguousarray(x_array, dtype=np.float32)).to(torch.float32)
		y_tensor = torch.from_numpy(np.ascontiguousarray(y_array, dtype=np.float32)).to(torch.float32)
		if not self.return_metadata:
			return x_tensor, y_tensor

		metadata = dict(self.metadata[index]) if self.metadata else {}
		metadata["cache_shard_path"] = str(self.shards[shard_index]["path"])
		metadata["cache_local_index"] = int(local_index)
		return x_tensor, y_tensor, metadata
