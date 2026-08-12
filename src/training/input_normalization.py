"""Shared input-normalization helpers for training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np

try:
	import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - torch is optional for pure metadata checks
	torch = None

from src.data.preprocessing import load_normalization_stats, resolve_input_normalization_device
from src.data.cache import compute_dataset_index_hash


def _section(config: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
	if not isinstance(config, Mapping):
		return {}
	value = config.get(name)
	return dict(value) if isinstance(value, Mapping) else {}


def _resolve_path(config: Mapping[str, Any] | None, configured_path: str | Path) -> Path:
	path = Path(configured_path).expanduser()
	if path.is_absolute():
		return path.resolve()
	config_path_value = None if not isinstance(config, Mapping) else config.get("config_path", config.get("_config_path"))
	if config_path_value:
		return (Path(config_path_value).expanduser().resolve().parent / path).resolve()
	return path.resolve()


def normalization_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
	"""Return the normalization config with backward-compatible aliases applied."""

	section = _section(config, "normalization")
	if "stats_path" not in section and "path" in section:
		section["stats_path"] = section["path"]
	if "path" not in section and "stats_path" in section:
		section["path"] = section["stats_path"]
	return section


def normalization_enabled(config: Mapping[str, Any] | None) -> bool:
	"""Return whether input normalization is configured as active."""

	section = normalization_config(config)
	if not section:
		return False
	if bool(section.get("enabled", True)) is False:
		return False
	return resolve_input_normalization_device(config) != "none"


def resolve_input_normalization_stats_path(config: Mapping[str, Any] | None, must_exist: bool = False) -> Path | None:
	"""Resolve the configured train-split normalization stats path."""

	section = normalization_config(config)
	configured_path = section.get("stats_path", section.get("path"))
	if configured_path in (None, "", "null"):
		if must_exist:
			raise FileNotFoundError("Normalization is enabled, but normalization.path/stats_path is not configured.")
		return None
	resolved = _resolve_path(config, configured_path)
	if not resolved.exists():
		if must_exist:
			raise FileNotFoundError(
				f"Normalization stats file not found: {resolved}\n"
				"Compute train-only input stats first, for example:\n"
				"python scripts/compute_normalization.py --config configs/default.yaml --from_cache"
			)
		return None
	return resolved


def load_input_normalization_stats(config: Mapping[str, Any], required: bool | None = None) -> dict[str, Any] | None:
	"""Load configured input-normalization stats, returning ``None`` when disabled/missing."""

	if not normalization_enabled(config):
		return None
	section = normalization_config(config)
	require_stats = bool(section.get("require_stats", False)) if required is None else bool(required)
	path = resolve_input_normalization_stats_path(config, must_exist=require_stats)
	if path is None:
		return None
	return load_normalization_stats(path)


def validate_normalization_stats(stats: Mapping[str, Any], input_channels: int, config: Mapping[str, Any] | None = None) -> None:
	"""Validate stats shape and train-only metadata before applying them."""

	if "mean" not in stats or "std" not in stats:
		raise KeyError("Input normalization stats must contain mean and std.")
	mean = np.asarray(stats["mean"]).reshape(-1)
	std = np.asarray(stats["std"]).reshape(-1)
	if mean.shape != std.shape:
		raise ValueError(f"Normalization mean/std shapes differ: mean={mean.shape}, std={std.shape}.")
	if int(mean.shape[0]) != int(input_channels):
		raise ValueError(
			"Normalization stats channel count does not match model inputs. "
			f"Expected {int(input_channels)}, got mean={mean.shape[0]} std={std.shape[0]}."
		)
	if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
		raise ValueError("Normalization stats contain NaN or Inf values.")
	if np.any(std <= 0.0):
		raise ValueError("Normalization std contains zero or negative values.")

	fit_split = stats.get("fit_split", stats.get("split_used", stats.get("source_split", None)))
	if fit_split is not None:
		fit_split_text = str(np.asarray(fit_split).reshape(-1)[0])
		allow_val_test = bool(normalization_config(config).get("allow_val_test_fit", False))
		if fit_split_text.lower() != "train" and not allow_val_test:
			raise ValueError(
				"Normalization stats were not marked as train-only. "
				f"fit_split={fit_split_text!r}; set normalization.allow_val_test_fit=true only for intentional diagnostics."
			)

	if isinstance(config, Mapping):
		normalization = normalization_config(config)
		allow_mismatch = bool(normalization.get("allow_normalization_mismatch", normalization.get("allow_mismatch", False)))
		stats_cache_version = stats.get("cache_version")
		if stats_cache_version is not None:
			stats_cache_version_text = str(np.asarray(stats_cache_version).reshape(-1)[0])
			cache_config = _section(config, "cache")
			current_cache_version = cache_config.get("cache_version")
			if current_cache_version not in (None, "", "null") and str(current_cache_version) != stats_cache_version_text:
				message = (
					"Normalization cache_version does not match current config: "
					f"stats={stats_cache_version_text!r}, config={str(current_cache_version)!r}."
				)
				if allow_mismatch:
					warnings.warn(message, RuntimeWarning, stacklevel=2)
				else:
					raise ValueError(message + " Set normalization.allow_normalization_mismatch=true only for intentional compatibility/debug runs.")
		stats_dataset_hash = stats.get("dataset_index_hash")
		if stats_dataset_hash is not None:
			stats_dataset_hash_text = str(np.asarray(stats_dataset_hash).reshape(-1)[0])
			current_dataset_hash = compute_dataset_index_hash(config)
			if current_dataset_hash not in (None, "", "null") and str(current_dataset_hash) != stats_dataset_hash_text:
				message = (
					"Normalization dataset_index_hash does not match current config: "
					f"stats={stats_dataset_hash_text!r}, config={str(current_dataset_hash)!r}."
				)
				if allow_mismatch:
					warnings.warn(message, RuntimeWarning, stacklevel=2)
				else:
					raise ValueError(message + " Set normalization.allow_normalization_mismatch=true only for intentional compatibility/debug runs.")


def should_apply_device_normalization(config: Mapping[str, Any] | None) -> bool:
	"""Return whether inputs should be normalized after transfer to the model device."""

	return normalization_enabled(config) and resolve_input_normalization_device(config) == "device"


def build_input_normalizer(
	config: Mapping[str, Any],
	device: Any = None,
	input_channels: int | None = None,
	stats: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
	"""Build cached device tensors for per-channel z-score input normalization."""

	if not should_apply_device_normalization(config):
		return None
	if torch is None:
		raise ImportError("PyTorch is required for device-side input normalization.")
	if input_channels is None:
		raise ValueError("input_channels is required when building a device input normalizer.")
	if stats is None:
		stats = load_input_normalization_stats(config, required=True)
	if stats is None:
		return None
	validate_normalization_stats(stats, int(input_channels), config)
	torch_device = torch.device("cpu" if device is None else device)
	mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=torch_device).flatten()
	std = torch.as_tensor(stats["std"], dtype=torch.float32, device=torch_device).flatten().clamp_min_(1.0e-6)
	return {
		"mean": mean.reshape(1, 1, int(input_channels), 1, 1),
		"std": std.reshape(1, 1, int(input_channels), 1, 1),
		"input_channels": int(input_channels),
		"device": str(torch_device),
	}


def build_input_normalizer_for_loader(loader, device: Any, input_channels: int, config: Mapping[str, Any] | None = None):
	"""Build a normalizer only when the loader dataset returns raw inputs for device normalization."""

	dataset = getattr(loader, "dataset", None)
	if not bool(getattr(dataset, "input_normalization_on_device", False)):
		return None
	stats = getattr(dataset, "normalization_stats", None)
	if not isinstance(stats, Mapping):
		if config is None:
			return None
		stats = load_input_normalization_stats(config, required=True)
	if config is None:
		config = getattr(dataset, "config", {}) if dataset is not None else {}
	validate_normalization_stats(stats, int(input_channels), config)
	return build_input_normalizer(config, device=device, input_channels=int(input_channels), stats=stats)


def apply_input_normalization(x, normalizer, config: Mapping[str, Any] | None = None):
	"""Apply a prepared normalizer in-place to a ``(B, T, C, H, W)`` tensor."""

	if normalizer is None:
		return x
	if torch is None or not torch.is_tensor(x):
		raise TypeError("apply_input_normalization expects a torch Tensor input.")
	if int(x.shape[2]) != int(normalizer.get("input_channels", x.shape[2])):
		raise ValueError(
			"Input tensor channel count does not match the prepared normalizer. "
			f"x channels={int(x.shape[2])}, normalizer channels={normalizer.get('input_channels')}."
		)
	with torch.no_grad():
		x.sub_(normalizer["mean"])
		x.div_(normalizer["std"])
	return x


def input_normalization_status(loader) -> str:
	"""Return a compact status string for a DataLoader's input normalization path."""

	dataset = getattr(loader, "dataset", None)
	if bool(getattr(dataset, "input_normalization_on_device", False)):
		return "device"
	if bool(getattr(dataset, "inputs_are_normalized", False)):
		return "dataset"
	if getattr(dataset, "normalization_stats", None) is None:
		return "none"
	return "unknown"


def normalization_metadata_from_loader(
	loader,
	config: Mapping[str, Any] | None,
	input_channels: int,
	stats_path: str | Path | None = None,
) -> dict[str, Any]:
	"""Return serializable metadata describing the active input-normalization path."""

	dataset = getattr(loader, "dataset", None)
	stats = getattr(dataset, "normalization_stats", None)
	stats_shapes: dict[str, list[int]] = {}
	channel_count = None
	if isinstance(stats, Mapping):
		for key in ("mean", "std", "min", "max"):
			if key in stats:
				stats_shapes[key] = list(np.asarray(stats[key]).shape)
		if "mean" in stats:
			channel_count = int(np.asarray(stats["mean"]).reshape(-1).shape[0])
	if stats_path is None:
		resolved = resolve_input_normalization_stats_path(config, must_exist=False)
		stats_path = str(resolved) if resolved is not None else None
	return {
		"enabled": normalization_enabled(config),
		"configured_device": resolve_input_normalization_device(config),
		"applied_by": input_normalization_status(loader),
		"stats_path": str(stats_path) if stats_path not in (None, "") else None,
		"stats_channel_count": channel_count,
		"input_channels": int(input_channels),
		"channel_count_matches": channel_count == int(input_channels) if channel_count is not None else None,
		"stats_shapes": stats_shapes,
	}


def compare_normalization_metadata(
	checkpoint_metadata: Mapping[str, Any] | None,
	current_metadata: Mapping[str, Any],
) -> list[str]:
	"""Return human-readable mismatches between checkpoint and current normalization metadata."""

	if not isinstance(checkpoint_metadata, Mapping) or not checkpoint_metadata:
		return []
	mismatches: list[str] = []
	for key in ("enabled", "configured_device", "applied_by", "stats_channel_count", "input_channels"):
		if key not in checkpoint_metadata or key not in current_metadata:
			continue
		if checkpoint_metadata.get(key) != current_metadata.get(key):
			mismatches.append(f"{key}: checkpoint={checkpoint_metadata.get(key)!r}, current={current_metadata.get(key)!r}")
	return mismatches


def input_batch_summary(x, prefix: str = "x") -> dict[str, float | str]:
	"""Summarize one tensor for before/after normalization diagnostics."""

	if torch is not None and torch.is_tensor(x):
		array = x.detach().float().cpu().numpy()
	else:
		array = np.asarray(x, dtype=np.float32)
	finite = array[np.isfinite(array)]
	if finite.size == 0:
		return {f"{prefix}_min": float("nan"), f"{prefix}_mean": float("nan"), f"{prefix}_max": float("nan"), f"{prefix}_std": float("nan")}
	return {
		f"{prefix}_min": float(finite.min()),
		f"{prefix}_mean": float(finite.mean()),
		f"{prefix}_max": float(finite.max()),
		f"{prefix}_std": float(finite.std()),
	}


def write_normalization_sidecar(path: str | Path, metadata: Mapping[str, Any]) -> Path:
	"""Write a JSON sidecar next to a stats archive."""

	sidecar_path = Path(path).expanduser().resolve().with_suffix(".json")
	sidecar_path.parent.mkdir(parents=True, exist_ok=True)
	with sidecar_path.open("w", encoding="utf-8") as handle:
		json.dump({str(key): value for key, value in metadata.items()}, handle, indent=2, sort_keys=True, default=str)
	return sidecar_path
