"""Visualize patchified samples from the precomputed wildfire patch cache."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

if "MPLCONFIGDIR" not in os.environ:
	_mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "fire_forecasting_mplconfig"
	_mpl_config_dir.mkdir(parents=True, exist_ok=True)
	os.environ["MPLCONFIGDIR"] = str(_mpl_config_dir)
if "XDG_CACHE_HOME" not in os.environ:
	_xdg_cache_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "fire_forecasting_xdg_cache"
	_xdg_cache_dir.mkdir(parents=True, exist_ok=True)
	os.environ["XDG_CACHE_HOME"] = str(_xdg_cache_dir)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.cache import MANIFEST_FILENAME, get_patch_cache_dir, load_cache_manifest
from src.data.preprocessing import load_normalization_stats, normalize_tensor
from src.data.stored_npz import open_stored_npz_array
from src.training.input_normalization import resolve_input_normalization_stats_path


TARGET_CHANNELS = [
	(0, "target surface consumed"),
	(1, "target canopy consumed"),
	(2, "target fire/perimeter mask"),
	(3, "target log1p energy"),
]


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Visualize one precomputed patch-cache sample at a time.")
	parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
	parser.add_argument("--split", choices=("train", "val", "test"), default="train", help="Cached split to inspect.")
	parser.add_argument("--cache_dir", default=None, help="Optional patch-cache directory override.")
	parser.add_argument("--sample_index", type=int, default=None, help="Global sample index within split cache.")
	parser.add_argument("--shard_index", type=int, default=None, help="Shard index to inspect.")
	parser.add_argument("--sample_in_shard", type=int, default=None, help="Sample offset inside --shard_index.")
	parser.add_argument("--random", action="store_true", help="Start at a random sample.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed.")
	parser.add_argument("--mode", choices=("interactive", "save"), default="interactive", help="Interactive viewer or batch screenshot mode.")
	parser.add_argument("--output_dir", default="artifacts/patch_cache_visualizations", help="Directory for saved screenshots.")
	parser.add_argument("--num_save_samples", type=int, default=10, help="Number of samples for --mode save.")
	parser.add_argument("--show_normalized", action="store_true", help="Apply configured input normalization to X before visualization.")
	parser.add_argument("--channel_manifest", default=None, help="Optional channel manifest JSON path.")
	parser.add_argument("--max_channels_per_group", type=int, default=9, help="Max input channels shown per grid page.")
	parser.add_argument("--dpi", type=int, default=140, help="Figure DPI.")
	parser.add_argument("--cmap", default="viridis", help="Default image colormap.")
	parser.add_argument("--error_cmap", default="magma", help="Colormap for energy/error-like target panels.")
	parser.add_argument("--robust_percentile", type=float, default=99.0, help="Upper robust percentile for display scaling.")
	return parser


def _ensure_config_path(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
	resolved = Path(config_path).expanduser().resolve()
	config = dict(config)
	config["config_path"] = str(resolved)
	config["_config_path"] = str(resolved)
	return config


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
				raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
			rows.append(value if isinstance(value, dict) else {"value": value})
	return rows


def _resolve_path(base: Path, value: str | Path | None) -> Path | None:
	if value in (None, "", "null"):
		return None
	path = Path(str(value)).expanduser()
	if path.is_absolute():
		return path.resolve()
	return (base.parent / path).resolve()


def _candidate_channel_manifest(config: Mapping[str, Any], config_path: Path, override: str | None) -> Path | None:
	if override not in (None, "", "null"):
		return _resolve_path(config_path, override)
	for section_name in ("channels", "channel_manifest", "metadata"):
		section = config.get(section_name)
		if isinstance(section, Mapping):
			for key in ("manifest", "path", "channel_manifest"):
				if section.get(key) not in (None, "", "null"):
					return _resolve_path(config_path, str(section[key]))
	for candidate in (PROJECT_ROOT / "artifacts" / "channel_manifest.json", config_path.parent / "channel_manifest.json"):
		if candidate.exists():
			return candidate.resolve()
	return None


def _load_channel_names(config: Mapping[str, Any], config_path: Path, override: str | None) -> dict[int, str]:
	path = _candidate_channel_manifest(config, config_path, override)
	if path is None or not path.exists():
		return {}
	try:
		payload = json.loads(path.read_text(encoding="utf-8"))
	except Exception as exc:
		print(f"WARNING: could not read channel manifest {path}: {exc}")
		return {}
	names: dict[int, str] = {}
	if isinstance(payload, Mapping):
		items = payload.get("channels", payload.get("input_channels", payload))
		if isinstance(items, list):
			for index, item in enumerate(items):
				if isinstance(item, Mapping):
					channel_index = int(item.get("index", item.get("channel", index)))
					names[channel_index] = str(item.get("name", item.get("label", f"ch{channel_index:03d}")))
				else:
					names[index] = str(item)
		elif isinstance(items, Mapping):
			for key, value in items.items():
				try:
					channel_index = int(str(key).replace("ch", ""))
				except ValueError:
					continue
				names[channel_index] = str(value.get("name", value.get("label")) if isinstance(value, Mapping) else value)
	return names


class PatchCacheReader:
	"""Lazy direct reader for split-level patch-cache shards."""

	def __init__(self, cache_dir: Path, split: str, manifest: Mapping[str, Any]) -> None:
		self.cache_dir = cache_dir.expanduser().resolve()
		self.split = split
		self.manifest = dict(manifest)
		shards_by_split = manifest.get("shards", {})
		if not isinstance(shards_by_split, Mapping) or split not in shards_by_split:
			raise ValueError(f"Manifest does not contain shards for split={split!r}.")
		raw_shards = shards_by_split[split]
		if not isinstance(raw_shards, list) or not raw_shards:
			raise ValueError(f"Manifest contains no shard entries for split={split!r}.")
		self.shards: list[dict[str, Any]] = []
		self.offsets: list[int] = [0]
		for shard_index, entry in enumerate(raw_shards):
			if isinstance(entry, Mapping):
				path_value = entry.get("path")
				num_samples = entry.get("num_samples")
			else:
				path_value = entry
				num_samples = None
			if path_value in (None, "", "null"):
				raise ValueError(f"Shard entry is missing path: {entry!r}")
			shard_path = Path(str(path_value)).expanduser()
			if not shard_path.is_absolute():
				shard_path = self.cache_dir / shard_path
			shard_path = shard_path.resolve()
			if num_samples is None:
				num_samples = self._shard_sample_count(shard_path)
			self.shards.append({"index": shard_index, "path": shard_path, "num_samples": int(num_samples)})
			self.offsets.append(self.offsets[-1] + int(num_samples))
		self.length = int(self.offsets[-1])
		self.metadata = _read_jsonl(self.cache_dir / split / "metadata.jsonl")
		if self.metadata and len(self.metadata) != self.length:
			print(f"WARNING: metadata rows={len(self.metadata)} but cached samples={self.length}.")
		self._open_shard_index: int | None = None
		self._open_shard: dict[str, Any] | None = None

	def _shard_sample_count(self, path: Path) -> int:
		if path.suffix.lower() == ".npz":
			with np.load(path, allow_pickle=False) as data:
				if "X" not in data.files or "y" not in data.files:
					raise ValueError(f"Shard {path} missing X/y. Available keys: {data.files}")
				return int(data["X"].shape[0])
		if path.suffix.lower() == ".pt":
			try:
				import torch
			except ImportError as exc:
				raise ImportError("PyTorch is required to read .pt cache shards.") from exc
			raw = torch.load(path, map_location="cpu")
			return int(raw["X"].shape[0])
		raise ValueError(f"Unsupported shard format: {path}")

	def locate(self, global_index: int) -> tuple[int, int]:
		index = int(global_index)
		if index < 0:
			index += self.length
		if index < 0 or index >= self.length:
			raise IndexError(f"sample_index={global_index} outside [0, {self.length - 1}]")
		shard_index = bisect_right(self.offsets, index) - 1
		return shard_index, index - self.offsets[shard_index]

	def global_from_shard(self, shard_index: int, sample_in_shard: int) -> int:
		if shard_index < 0 or shard_index >= len(self.shards):
			raise IndexError(f"shard_index={shard_index} outside [0, {len(self.shards) - 1}]")
		count = int(self.shards[shard_index]["num_samples"])
		if sample_in_shard < 0 or sample_in_shard >= count:
			raise IndexError(f"sample_in_shard={sample_in_shard} outside [0, {count - 1}]")
		return int(self.offsets[shard_index] + sample_in_shard)

	def _load_shard(self, shard_index: int) -> dict[str, Any]:
		if self._open_shard_index == shard_index and self._open_shard is not None:
			return self._open_shard
		shard_path = Path(self.shards[shard_index]["path"])
		if shard_path.suffix.lower() == ".npz":
			try:
				shard = {"X": open_stored_npz_array(shard_path, "X"), "y": open_stored_npz_array(shard_path, "y")}
			except Exception:
				archive = np.load(shard_path, allow_pickle=False)
				if "X" not in archive.files or "y" not in archive.files:
					raise ValueError(f"Shard {shard_path} missing X/y. Available keys: {archive.files}")
				shard = {"X": archive["X"], "y": archive["y"], "_archive": archive}
		elif shard_path.suffix.lower() == ".pt":
			try:
				import torch
			except ImportError as exc:
				raise ImportError("PyTorch is required to read .pt cache shards.") from exc
			raw = torch.load(shard_path, map_location="cpu")
			x = raw["X"].detach().cpu().numpy() if hasattr(raw["X"], "detach") else np.asarray(raw["X"])
			y = raw["y"].detach().cpu().numpy() if hasattr(raw["y"], "detach") else np.asarray(raw["y"])
			shard = {"X": x, "y": y}
		else:
			raise ValueError(f"Unsupported shard format: {shard_path}")
		self._open_shard_index = shard_index
		self._open_shard = shard
		return shard

	def get(self, global_index: int) -> dict[str, Any]:
		shard_index, local_index = self.locate(global_index)
		shard = self._load_shard(shard_index)
		x = np.asarray(shard["X"][local_index], dtype=np.float32)
		y = np.asarray(shard["y"][local_index], dtype=np.float32)
		metadata = dict(self.metadata[global_index]) if self.metadata and global_index < len(self.metadata) else {}
		metadata.setdefault("cache_shard_path", str(self.shards[shard_index]["path"]))
		metadata.setdefault("cache_local_index", int(local_index))
		metadata.setdefault("cache_global_index", int(global_index))
		return {"X": x, "y": y, "metadata": metadata, "shard_index": shard_index, "local_index": local_index}


def _normalization_stats(config: Mapping[str, Any], show_normalized: bool) -> dict[str, np.ndarray] | None:
	if not show_normalized:
		return None
	path = resolve_input_normalization_stats_path(config, must_exist=True)
	if path is None:
		raise FileNotFoundError("Normalization was requested but no stats path is configured.")
	return load_normalization_stats(path)


def _apply_normalization(x: np.ndarray, stats: Mapping[str, Any] | None) -> np.ndarray:
	if stats is None:
		return x
	mean = np.asarray(stats["mean"], dtype=np.float32)
	std = np.asarray(stats["std"], dtype=np.float32)
	if mean.shape[0] != x.shape[1]:
		raise ValueError(f"Normalization channel mismatch: X has C={x.shape[1]}, stats mean has {mean.shape[0]}.")
	x_last = np.transpose(x, (0, 2, 3, 1))
	x_norm = normalize_tensor(x_last, mean, std)
	return np.ascontiguousarray(np.transpose(x_norm, (0, 3, 1, 2)), dtype=np.float32)


def _channel_groups(total_channels: int) -> list[tuple[str, list[int]]]:
	groups: list[tuple[str, list[int]]] = []
	core = [index for index in (80, 81, 82, 83, 84, 85) if index < total_channels]
	if core:
		groups.append(("core_fire_fuel", core))
	for level in range(8):
		start = level * 10
		channels = [index for index in range(start, min(start + 10, total_channels))]
		if channels:
			groups.append((f"atmospheric_level_{level}", channels))
	if total_channels > 86:
		groups.append(("engineered_features", list(range(86, total_channels))))
	all_channels = list(range(total_channels))
	groups.append(("all_channels", all_channels))
	return groups


def _default_channel_name(channel: int) -> str:
	known = {
		80: "surface sensible flux",
		81: "surface latent flux",
		82: "canopy sensible flux",
		83: "canopy latent flux",
		84: "surface fuel",
		85: "canopy fuel",
	}
	if channel in known:
		return known[channel]
	level = channel // 10
	offset = channel % 10
	if 0 <= channel <= 79:
		variables = ["U", "V", "W", "T", "QV", "QC", "QR", "QI", "QS", "P"]
		return f"z{level} {variables[offset] if offset < len(variables) else f'var{offset}'}"
	return f"ch{channel:03d}"


def _channel_name(channel_names: Mapping[int, str], channel: int) -> str:
	return channel_names.get(int(channel), _default_channel_name(int(channel)))


def _stats_line(name: str, array: np.ndarray) -> str:
	values = np.asarray(array, dtype=np.float32)
	finite = values[np.isfinite(values)]
	if finite.size == 0:
		return f"{name}: all NaN/Inf"
	return f"{name}: min={finite.min():.6g} max={finite.max():.6g} mean={finite.mean():.6g}"


def _patch_text(metadata: Mapping[str, Any]) -> str:
	patch = metadata.get("patch")
	if isinstance(patch, Mapping):
		return f"({patch.get('y0')}:{patch.get('y1')},{patch.get('x0')}:{patch.get('x1')})"
	top = metadata.get("patch_top")
	left = metadata.get("patch_left")
	bottom = metadata.get("patch_bottom")
	right = metadata.get("patch_right")
	size = metadata.get("patch_size")
	if bottom is None and top is not None and size is not None:
		bottom = int(top) + int(size)
	if right is None and left is not None and size is not None:
		right = int(left) + int(size)
	if top is None or left is None:
		return "metadata unavailable"
	return f"({top}:{bottom},{left}:{right})"


def _input_indices(metadata: Mapping[str, Any], input_sequence_length: int) -> list[int] | None:
	value = metadata.get("input_indices")
	if isinstance(value, list):
		try:
			return [int(item) for item in value]
		except (TypeError, ValueError):
			return None
	start = metadata.get("start_idx", metadata.get("sample_index"))
	try:
		start_idx = int(start)
	except (TypeError, ValueError):
		return None
	return list(range(start_idx, start_idx + input_sequence_length))


def _target_idx(metadata: Mapping[str, Any], input_indices: list[int] | None, horizon: int) -> int | None:
	for key in ("target_idx", "future_idx", "future_index", "target_index"):
		if metadata.get(key) is not None:
			try:
				return int(metadata[key])
			except (TypeError, ValueError):
				pass
	if input_indices:
		return int(input_indices[-1] + horizon)
	return None


def _sample_title(split: str, global_index: int, sample: Mapping[str, Any], input_sequence_length: int, horizon: int, normalized: bool) -> str:
	metadata = sample["metadata"]
	inputs = _input_indices(metadata, input_sequence_length)
	target = _target_idx(metadata, inputs, horizon)
	fire = metadata.get("fire_name", metadata.get("dataset_name", metadata.get("data_dir", "metadata unavailable")))
	prefix = "NORMALIZED INPUT VIEW | " if normalized else ""
	return (
		f"{prefix}split={split} | fire={fire} | sample={global_index} | shard={sample['shard_index']} local={sample['local_index']} | "
		f"patch={_patch_text(metadata)}\n"
		f"T={input_sequence_length} horizon={horizon} | inputs={inputs if inputs is not None else 'unknown'} -> target={target}"
	)


def _validate_sample(sample: Mapping[str, Any], input_sequence_length: int, horizon: int) -> None:
	x = sample["X"]
	y = sample["y"]
	metadata = sample["metadata"]
	if x.ndim != 4:
		print(f"WARNING: expected X.ndim == 4, got shape {x.shape}")
	if y.ndim != 3:
		print(f"WARNING: expected y.ndim == 3, got shape {y.shape}")
	if x.ndim >= 1 and int(x.shape[0]) != int(input_sequence_length):
		print(f"WARNING: X T={x.shape[0]} but config input_sequence_length={input_sequence_length}")
	if y.ndim >= 1 and int(y.shape[0]) != 4:
		print(f"WARNING: y channels={y.shape[0]} but expected 4")
	inputs = _input_indices(metadata, input_sequence_length)
	target = _target_idx(metadata, inputs, horizon)
	if inputs is not None and len(inputs) != int(input_sequence_length):
		print(f"WARNING: metadata input_indices length={len(inputs)} but expected {input_sequence_length}")
	if inputs is not None and target is not None and int(target) != int(inputs[-1] + horizon):
		print(f"WARNING: metadata target={target} but last_input+horizon={inputs[-1] + horizon}")
	if not np.isfinite(x).all():
		print("WARNING: X contains NaN or Inf values.")
	if not np.isfinite(y).all():
		print("WARNING: y contains NaN or Inf values.")


def _print_sample_console(split: str, global_index: int, sample: Mapping[str, Any], input_sequence_length: int, horizon: int) -> None:
	x = sample["X"]
	y = sample["y"]
	metadata = sample["metadata"]
	inputs = _input_indices(metadata, input_sequence_length)
	target = _target_idx(metadata, inputs, horizon)
	print("\n========== Patch Cache Sample ==========")
	print(f"sample global index: {global_index}")
	print(f"shard path: {metadata.get('cache_shard_path', 'metadata unavailable')}")
	print(f"sample in shard: {sample['local_index']}")
	print(f"fire name: {metadata.get('fire_name', metadata.get('dataset_name', 'metadata unavailable'))}")
	print(f"split: {metadata.get('split', split)}")
	print(f"patch coordinates: {_patch_text(metadata)}")
	print(f"input indices: {inputs if inputs is not None else 'metadata unavailable'}")
	print(f"last input index: {inputs[-1] if inputs else metadata.get('last_input_idx', 'metadata unavailable')}")
	print(f"target index: {target if target is not None else 'metadata unavailable'}")
	print(f"prediction horizon: {horizon}")
	print(f"X shape: {x.shape}")
	print(f"y shape: {y.shape}")
	print(_stats_line("X", x))
	for channel, name in TARGET_CHANNELS:
		if y.ndim == 3 and channel < y.shape[0]:
			print(_stats_line(f"y[{channel}] {name}", y[channel]))
	if y.ndim == 3 and y.shape[0] > 2:
		print(f"active pixels in target mask: {int(np.sum(y[2] > 0.5))}")
	if y.ndim == 3 and y.shape[0] > 0:
		print(f"nonzero target surface pixels: {int(np.sum(y[0] > 0.0))}")
	if y.ndim == 3 and y.shape[0] > 1:
		print(f"nonzero target canopy pixels: {int(np.sum(y[1] > 0.0))}")
	if y.ndim == 3 and y.shape[0] > 3:
		print(f"nonzero energy pixels: {int(np.sum(y[3] > 0.0))}")


def _display_limits(image: np.ndarray, robust_percentile: float, nonnegative: bool = False, mask: bool = False) -> tuple[float, float]:
	array = np.asarray(image, dtype=np.float32)
	finite = array[np.isfinite(array)]
	if finite.size == 0:
		return 0.0, 1.0
	if mask:
		return 0.0, 1.0
	low = 0.0 if nonnegative else float(np.nanpercentile(finite, 1.0))
	high = float(np.nanpercentile(finite, robust_percentile))
	if not np.isfinite(high) or high <= low:
		high = float(np.nanmax(finite))
	if not np.isfinite(high) or high <= low:
		high = low + 1.0
	return low, high


class PatchCacheViewer:
	def __init__(
		self,
		reader: PatchCacheReader,
		config: Mapping[str, Any],
		args: argparse.Namespace,
		channel_names: Mapping[int, str],
		normalization_stats: Mapping[str, Any] | None,
		start_index: int,
	) -> None:
		self.reader = reader
		self.config = config
		self.args = args
		self.channel_names = dict(channel_names)
		self.normalization_stats = normalization_stats
		self.index = int(start_index)
		self.input_sequence_length = int(config.get("input_sequence_length", reader.manifest.get("input_sequence_length", 5)))
		self.horizon = int(config.get("prediction_horizon", reader.manifest.get("prediction_horizon", 10)))
		total_channels = int(reader.manifest.get("input_channels", config.get("model", {}).get("input_channels", 0)))
		self.groups = _channel_groups(total_channels)
		self.group_index = 0
		self.channel_page = 0
		self.frame_index = 0
		self.selected_channel = min(84, max(0, total_channels - 1))
		self.views = ["input_grid", "time_sequence", "target_grid", "all_inputs_summary"]
		self.view = "target_grid"
		self.rng = np.random.default_rng(int(args.seed))
		self.fig = None
		self.axes = None
		self.current_sample: dict[str, Any] | None = None

	def _load_current(self) -> dict[str, Any]:
		raw = self.reader.get(self.index)
		x = _apply_normalization(raw["X"], self.normalization_stats)
		sample = dict(raw)
		sample["X"] = x
		_validate_sample(sample, self.input_sequence_length, self.horizon)
		self.frame_index = int(np.clip(self.frame_index, 0, max(0, x.shape[0] - 1)))
		self.current_sample = sample
		return sample

	def _panel(self, title: str, image: np.ndarray, cmap: str | None = None, nonnegative: bool = False, mask: bool = False) -> tuple[str, np.ndarray, str, bool, bool]:
		return title, np.asarray(image, dtype=np.float32), cmap or self.args.cmap, nonnegative, mask

	def _target_panels(self, sample: Mapping[str, Any]) -> list[tuple[str, np.ndarray, str, bool, bool]]:
		x = sample["X"]
		y = sample["y"]
		latest = x[-1]
		zero = np.zeros_like(y[0])
		def xch(channel: int) -> np.ndarray:
			return latest[channel] if channel < latest.shape[0] else zero
		panels = [
			self._panel("last input surface fuel X[-1,84]", xch(84), nonnegative=True),
			self._panel("last input canopy fuel X[-1,85]", xch(85), nonnegative=True),
			self._panel("last input total flux 80:83", sum(xch(ch) for ch in (80, 81, 82, 83)), nonnegative=True),
			self._panel("target surface consumed y[0]", y[0] if y.shape[0] > 0 else zero, nonnegative=True),
			self._panel("target canopy consumed y[1]", y[1] if y.shape[0] > 1 else zero, nonnegative=True),
			self._panel("target mask y[2]", y[2] if y.shape[0] > 2 else zero, mask=True),
			self._panel("target energy log y[3]", y[3] if y.shape[0] > 3 else zero, cmap=self.args.error_cmap, nonnegative=True),
			self._panel("estimated future surface fuel", xch(84) - (y[0] if y.shape[0] > 0 else zero), nonnegative=True),
			self._panel("estimated future canopy fuel", xch(85) - (y[1] if y.shape[0] > 1 else zero), nonnegative=True),
		]
		return panels

	def _input_grid_panels(self, sample: Mapping[str, Any]) -> list[tuple[str, np.ndarray, str, bool, bool]]:
		x = sample["X"]
		group_name, channels = self.groups[self.group_index]
		page_size = max(1, int(self.args.max_channels_per_group))
		start = self.channel_page * page_size
		selected = channels[start : start + page_size]
		if not selected:
			self.channel_page = 0
			selected = channels[:page_size]
		panels = []
		for channel in selected[:9]:
			if channel >= x.shape[1]:
				continue
			title = f"{group_name} | t={self.frame_index} | ch{channel:03d} {_channel_name(self.channel_names, channel)}"
			panels.append(self._panel(title, x[self.frame_index, channel], nonnegative=channel >= 80))
		return panels

	def _time_sequence_panels(self, sample: Mapping[str, Any]) -> list[tuple[str, np.ndarray, str, bool, bool]]:
		x = sample["X"]
		y = sample["y"]
		channel = int(np.clip(self.selected_channel, 0, x.shape[1] - 1))
		panels = [
			self._panel(f"t={time_index} ch{channel:03d} {_channel_name(self.channel_names, channel)}", x[time_index, channel], nonnegative=channel >= 80)
			for time_index in range(min(x.shape[0], 5))
		]
		if channel == 84 and y.shape[0] > 0:
			panels.append(self._panel("target surface consumed y[0]", y[0], nonnegative=True))
			panels.append(self._panel("estimated future surface fuel", x[-1, channel] - y[0], nonnegative=True))
		elif channel == 85 and y.shape[0] > 1:
			panels.append(self._panel("target canopy consumed y[1]", y[1], nonnegative=True))
			panels.append(self._panel("estimated future canopy fuel", x[-1, channel] - y[1], nonnegative=True))
		elif channel in (80, 81, 82, 83) and y.shape[0] > 3:
			panels.append(self._panel("target energy log y[3]", y[3], cmap=self.args.error_cmap, nonnegative=True))
		elif y.shape[0] > 2:
			panels.append(self._panel("target mask y[2]", y[2], mask=True))
		return panels

	def _summary_panels(self, sample: Mapping[str, Any]) -> list[tuple[str, np.ndarray, str, bool, bool]]:
		x = sample["X"]
		y = sample["y"]
		latest = x[-1]
		zero = np.zeros_like(y[0])
		def xch(channel: int) -> np.ndarray:
			return latest[channel] if channel < latest.shape[0] else zero
		low_wind = np.sqrt(np.square(xch(0)) + np.square(xch(1))) if latest.shape[0] > 1 else zero
		updraft = xch(2)
		engineered = xch(86) if latest.shape[0] > 86 else zero
		return [
			self._panel("last surface fuel ch84", xch(84), nonnegative=True),
			self._panel("last canopy fuel ch85", xch(85), nonnegative=True),
			self._panel("last total flux ch80-83", sum(xch(ch) for ch in (80, 81, 82, 83)), nonnegative=True),
			self._panel("low-level wind speed sqrt(U0^2+V0^2)", low_wind, nonnegative=True),
			self._panel("updraft proxy ch2", updraft),
			self._panel("engineered ch86" if latest.shape[0] > 86 else "engineered unavailable", engineered),
			self._panel("target surface consumed y[0]", y[0] if y.shape[0] > 0 else zero, nonnegative=True),
			self._panel("target mask y[2]", y[2] if y.shape[0] > 2 else zero, mask=True),
			self._panel("target energy log y[3]", y[3] if y.shape[0] > 3 else zero, cmap=self.args.error_cmap, nonnegative=True),
		]

	def _panels(self, sample: Mapping[str, Any]) -> list[tuple[str, np.ndarray, str, bool, bool]]:
		if self.view == "input_grid":
			return self._input_grid_panels(sample)
		if self.view == "time_sequence":
			return self._time_sequence_panels(sample)
		if self.view == "all_inputs_summary":
			return self._summary_panels(sample)
		return self._target_panels(sample)

	def render(self, *, print_console: bool = False):
		import matplotlib.pyplot as plt

		sample = self._load_current()
		if print_console:
			_print_sample_console(self.reader.split, self.index, sample, self.input_sequence_length, self.horizon)
		if self.fig is None:
			self.fig, self.axes = plt.subplots(3, 3, figsize=(13, 10), dpi=int(self.args.dpi), constrained_layout=True)
			self.fig.canvas.mpl_connect("key_press_event", self.on_key)
		assert self.axes is not None
		for axis in self.axes.ravel():
			axis.clear()
			axis.set_xticks([])
			axis.set_yticks([])
		for axis, panel in zip(self.axes.ravel(), self._panels(sample)):
			title, image, cmap, nonnegative, mask = panel
			vmin, vmax = _display_limits(image, float(self.args.robust_percentile), nonnegative=nonnegative, mask=mask)
			axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
			axis.set_title(title, fontsize=8)
		self.fig.suptitle(
			_sample_title(self.reader.split, self.index, sample, self.input_sequence_length, self.horizon, bool(self.args.show_normalized))
			+ f"\nview={self.view} | keys: h help, n/p sample, up/down frame, g/G group, c/C channel page, v view, w save, q quit",
			fontsize=9,
		)
		self.fig.canvas.draw_idle()
		return self.fig

	def save_current(self, output_dir: Path, prefix: str | None = None) -> Path:
		if self.fig is None:
			self.render(print_console=False)
		output_dir.mkdir(parents=True, exist_ok=True)
		name = prefix or f"{self.reader.split}_sample_{self.index:07d}_{self.view}.png"
		path = output_dir / name
		assert self.fig is not None
		self.fig.savefig(path, bbox_inches="tight")
		print(f"Saved figure: {path}")
		return path

	def _move_sample(self, delta: int) -> None:
		self.index = int(np.clip(self.index + delta, 0, self.reader.length - 1))
		self.render(print_console=True)

	def _random_sample(self) -> None:
		self.index = int(self.rng.integers(0, self.reader.length))
		self.render(print_console=True)

	def _jump_sample(self) -> None:
		try:
			value = input(f"Jump to global sample index [0, {self.reader.length - 1}]: ").strip()
			self.index = int(np.clip(int(value), 0, self.reader.length - 1))
		except Exception as exc:
			print(f"Invalid sample index: {exc}")
		self.render(print_console=True)

	def _cycle_view(self) -> None:
		index = (self.views.index(self.view) + 1) % len(self.views)
		self.view = self.views[index]
		self.render()

	def _move_group(self, delta: int) -> None:
		self.group_index = (self.group_index + delta) % len(self.groups)
		self.channel_page = 0
		self.render()

	def _move_channel_page(self, delta: int) -> None:
		group_name, channels = self.groups[self.group_index]
		page_size = max(1, int(self.args.max_channels_per_group))
		max_page = max(0, (len(channels) - 1) // page_size)
		self.channel_page = int(np.clip(self.channel_page + delta, 0, max_page))
		start = self.channel_page * page_size
		if channels[start : start + page_size]:
			self.selected_channel = channels[start]
		print(f"group={group_name} page={self.channel_page}/{max_page} selected_channel={self.selected_channel}")
		self.render()

	def _print_help(self) -> None:
		print(
			"""
Patch cache viewer controls
  n/right: next sample        p/left: previous sample
  r: random sample            j: jump to global sample index
  up/down: input timestamp    0-4: jump to input timestamp
  g/G: next/previous group    c/C: next/previous channel page
  v: cycle view               i: input_grid
  s: time_sequence            t: target_grid
  a: all_inputs_summary       w: save current figure
  h: print help               q/escape: quit
"""
		)

	def on_key(self, event) -> None:
		key = event.key
		if key in {"q", "escape"}:
			import matplotlib.pyplot as plt
			plt.close(self.fig)
			return
		if key in {"n", "right"}:
			self._move_sample(1)
		elif key in {"p", "left"}:
			self._move_sample(-1)
		elif key == "r":
			self._random_sample()
		elif key == "j":
			self._jump_sample()
		elif key == "up":
			self.frame_index = min(self.frame_index + 1, self.input_sequence_length - 1)
			self.render()
		elif key == "down":
			self.frame_index = max(self.frame_index - 1, 0)
			self.render()
		elif key in {"0", "1", "2", "3", "4"}:
			self.frame_index = min(int(key), self.input_sequence_length - 1)
			self.render()
		elif key == "g":
			self._move_group(1)
		elif key == "G":
			self._move_group(-1)
		elif key == "c":
			self._move_channel_page(1)
		elif key == "C":
			self._move_channel_page(-1)
		elif key == "v":
			self._cycle_view()
		elif key == "i":
			self.view = "input_grid"
			self.render()
		elif key == "s":
			self.view = "time_sequence"
			self.render()
		elif key == "t":
			self.view = "target_grid"
			self.render()
		elif key == "a":
			self.view = "all_inputs_summary"
			self.render()
		elif key == "w":
			self.save_current(Path(self.args.output_dir).expanduser().resolve())
		elif key == "h":
			self._print_help()


def _start_index(args: argparse.Namespace, reader: PatchCacheReader, rng: np.random.Generator) -> int:
	if args.shard_index is not None or args.sample_in_shard is not None:
		if args.shard_index is None or args.sample_in_shard is None:
			raise ValueError("--shard_index and --sample_in_shard must be provided together.")
		return reader.global_from_shard(int(args.shard_index), int(args.sample_in_shard))
	if args.sample_index is not None:
		return int(args.sample_index)
	if bool(args.random):
		return int(rng.integers(0, reader.length))
	return 0


def _save_mode(viewer: PatchCacheViewer, output_root: Path, num_samples: int, randomize: bool) -> None:
	import matplotlib.pyplot as plt

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	output_dir = output_root / f"{viewer.reader.split}_{timestamp}"
	output_dir.mkdir(parents=True, exist_ok=True)
	viewer.view = "target_grid"
	if randomize:
		indices = viewer.rng.choice(viewer.reader.length, size=min(num_samples, viewer.reader.length), replace=False).tolist()
	else:
		indices = list(range(viewer.index, min(viewer.index + int(num_samples), viewer.reader.length)))
	for output_index, sample_index in enumerate(indices):
		viewer.index = int(sample_index)
		viewer.render(print_console=True)
		viewer.save_current(output_dir, prefix=f"sample_{output_index:03d}_target_grid.png")
	plt.close(viewer.fig)
	print(f"Saved {len(indices)} patch-cache visualization(s) under {output_dir}")


def main() -> None:
	args = build_arg_parser().parse_args()
	if args.mode == "save":
		import matplotlib
		matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt

	config_path = Path(args.config).expanduser().resolve()
	config = _ensure_config_path(load_config(config_path), config_path)
	cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else get_patch_cache_dir(config)
	manifest_path = cache_dir / MANIFEST_FILENAME
	if not manifest_path.exists():
		raise FileNotFoundError(f"Patch-cache manifest not found: {manifest_path}")
	manifest = load_cache_manifest(cache_dir)
	reader = PatchCacheReader(cache_dir, args.split, manifest)
	if reader.length <= 0:
		raise ValueError(f"No cached samples for split={args.split!r} in {cache_dir}")
	channel_names = _load_channel_names(config, config_path, args.channel_manifest)
	normalization_stats = _normalization_stats(config, bool(args.show_normalized))
	rng = np.random.default_rng(int(args.seed))
	start_index = _start_index(args, reader, rng)
	viewer = PatchCacheViewer(reader, config, args, channel_names, normalization_stats, start_index)
	print(f"cache_dir: {cache_dir}")
	print(f"manifest: {manifest_path}")
	print(f"split: {args.split} samples={reader.length} shards={len(reader.shards)}")
	print(f"cache input: T={manifest.get('input_sequence_length')} C={manifest.get('input_channels')} horizon={manifest.get('prediction_horizon')}")
	if args.show_normalized:
		print("NORMALIZED INPUT VIEW enabled; y targets remain unchanged.")

	if args.mode == "save":
		_save_mode(viewer, Path(args.output_dir).expanduser().resolve(), int(args.num_save_samples), bool(args.random))
		return

	viewer.render(print_console=True)
	viewer._print_help()
	plt.show()


if __name__ == "__main__":
	main()
