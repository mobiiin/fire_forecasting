"""Leakage-safe deterministic baselines for the canonical Table 2 dataset."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.baselines.common import build_mask_logits_from_binary
from src.data.fire_mask_thresholds import threshold_union_mask
from src.data.processed_targets import build_processed_target


FrameLoader = Callable[[Path], np.ndarray]


def _load_raw_npz(path: Path) -> np.ndarray:
	with np.load(path, allow_pickle=False) as archive:
		if "x_raw" not in archive.files:
			raise KeyError(f"Processed frame does not contain x_raw: {path}")
		return np.asarray(archive["x_raw"], dtype=np.float32)


def _crop_chw(array: np.ndarray, patch: Mapping[str, Any]) -> np.ndarray:
	y0, x0, height, width = (int(patch[key]) for key in ("y0", "x0", "height", "width"))
	if y0 < 0 or x0 < 0 or height <= 0 or width <= 0:
		raise ValueError(f"Invalid processed patch: {dict(patch)}")
	if y0 + height > array.shape[-2] or x0 + width > array.shape[-1]:
		raise ValueError(f"Patch {dict(patch)} is outside array shape={array.shape}.")
	return np.asarray(array[..., y0 : y0 + height, x0 : x0 + width], dtype=np.float32)


def load_observed_raw_history(
	dataset_root: str | Path,
	record: Mapping[str, Any],
	*,
	frame_loader: FrameLoader = _load_raw_npz,
) -> list[np.ndarray]:
	"""Load only the frame indices explicitly listed as model inputs.

	The target path and target index are deliberately ignored.  This narrow API
	is the anti-leakage boundary used by both deterministic Table 2 baselines.
	"""

	root = Path(dataset_root).expanduser().resolve()
	fire_name = str(record["fire_name"])
	input_indices = [int(value) for value in record["input_indices"]]
	if len(input_indices) < 2 or input_indices != sorted(input_indices) or len(set(input_indices)) != len(input_indices):
		raise ValueError(f"Observed input_indices must be unique and chronological: {input_indices}")
	current_index = int(record["current_index"])
	if input_indices[-1] != current_index:
		raise ValueError(f"Last observed input {input_indices[-1]} does not equal current_index={current_index}.")
	if "target_index" in record and int(record["target_index"]) <= current_index:
		raise ValueError("Table 2 record target_index must be strictly after the final observed input.")
	frames: list[np.ndarray] = []
	for frame_index in input_indices:
		path = root / "fires" / fire_name / "frames" / f"frame_{frame_index:06d}.npz"
		frame = np.asarray(frame_loader(path), dtype=np.float32)
		if frame.ndim != 3 or int(frame.shape[0]) < 86:
			raise ValueError(f"Expected raw C,H,W frame with at least 86 channels, got {frame.shape}: {path}")
		if not np.isfinite(frame).all():
			raise ValueError(f"Observed raw frame contains NaN/Inf: {path}")
		frames.append(frame)
	return frames


def _target_thresholds(config: Mapping[str, Any]) -> dict[str, float]:
	target = config.get("target_construction", {}) if isinstance(config.get("target_construction"), Mapping) else {}
	mask = target.get("fire_mask", {}) if isinstance(target.get("fire_mask"), Mapping) else {}
	keys = ("energy_threshold_mw", "surface_fuel_threshold", "canopy_fuel_threshold")
	missing = [key for key in keys if key not in mask]
	if missing:
		raise KeyError(f"Canonical target-construction thresholds are missing: {missing}")
	return {key: float(mask[key]) for key in keys}


def _chw_to_hwc(frame: np.ndarray) -> np.ndarray:
	return np.transpose(np.asarray(frame[:86], dtype=np.float32), (1, 2, 0))


def persistence_from_observed_history(
	frames: Sequence[np.ndarray],
	area_2d: np.ndarray,
	config: Mapping[str, Any],
) -> np.ndarray:
	"""Persist the latest fully observed interval target from ``t-10 -> t``."""

	if len(frames) < 2:
		raise ValueError("Persistence requires at least two observed frames.")
	latest = build_processed_target(
		_chw_to_hwc(frames[-2]),
		_chw_to_hwc(frames[-1]),
		np.asarray(area_2d, dtype=np.float32),
		config,
		thresholds=_target_thresholds(config),
	)
	return np.stack(
		[
			latest["surface_consumed"],
			latest["canopy_consumed"],
			build_mask_logits_from_binary(latest["fire_mask"]),
			latest["energy_log"],
		],
		axis=0,
	).astype(np.float32, copy=False)


def linear_extrapolation_from_observed_history(
	frames: Sequence[np.ndarray],
	area_2d: np.ndarray,
	config: Mapping[str, Any],
) -> np.ndarray:
	"""Equal-step extrapolation from the two latest observed interval targets."""

	if len(frames) < 3:
		return persistence_from_observed_history(frames, area_2d, config)
	thresholds = _target_thresholds(config)
	previous = build_processed_target(
		_chw_to_hwc(frames[-3]),
		_chw_to_hwc(frames[-2]),
		np.asarray(area_2d, dtype=np.float32),
		config,
		thresholds=thresholds,
	)
	latest = build_processed_target(
		_chw_to_hwc(frames[-2]),
		_chw_to_hwc(frames[-1]),
		np.asarray(area_2d, dtype=np.float32),
		config,
		thresholds=thresholds,
	)
	current = _chw_to_hwc(frames[-1])
	surface = np.maximum(2.0 * latest["surface_consumed"] - previous["surface_consumed"], 0.0)
	canopy = np.maximum(2.0 * latest["canopy_consumed"] - previous["canopy_consumed"], 0.0)
	surface = np.minimum(surface, np.maximum(current[:, :, 84], 0.0)).astype(np.float32, copy=False)
	canopy = np.minimum(canopy, np.maximum(current[:, :, 85], 0.0)).astype(np.float32, copy=False)
	energy_mw = np.maximum(
		2.0 * latest["energy_release_mw"] - previous["energy_release_mw"],
		0.0,
	).astype(np.float32, copy=False)
	mask = threshold_union_mask(energy_mw, surface, canopy, thresholds)
	return np.stack(
		[
			surface,
			canopy,
			build_mask_logits_from_binary(mask),
			np.log1p(energy_mw).astype(np.float32, copy=False),
		],
		axis=0,
	).astype(np.float32, copy=False)


class ProcessedHistoryBaselinePredictor:
	"""Batch predictor that can access observed metadata but never targets."""

	def __init__(
		self,
		method: str,
		dataset_root: str | Path,
		config: Mapping[str, Any],
		*,
		frame_loader: FrameLoader = _load_raw_npz,
	) -> None:
		method = str(method).lower()
		if method not in {"persistence", "linear_extrapolation"}:
			raise ValueError(f"Unsupported deterministic Table 2 method: {method!r}")
		self.method = method
		self.dataset_root = Path(dataset_root).expanduser().resolve()
		self.config = config
		self.frame_loader = frame_loader

	@lru_cache(maxsize=32)
	def _full_prediction(self, fire_name: str, input_indices: tuple[int, ...], current_index: int) -> np.ndarray:
		# The published formulas only consume the latest observed interval
		# (two frames) or latest two intervals (three frames). Loading earlier
		# sequence frames is redundant and makes CPU evaluation needlessly costly.
		required_indices = input_indices[-2:] if self.method == "persistence" else input_indices[-3:]
		record = {
			"fire_name": fire_name,
			"input_indices": list(required_indices),
			"current_index": current_index,
		}
		frames = load_observed_raw_history(
			self.dataset_root,
			record,
			frame_loader=self.frame_loader,
		)
		area_path = self.dataset_root / "fires" / fire_name / "geometry" / "area_2d.npy"
		area = np.asarray(np.load(area_path, allow_pickle=False), dtype=np.float32)
		if self.method == "persistence":
			return persistence_from_observed_history(frames, area, self.config)
		return linear_extrapolation_from_observed_history(frames, area, self.config)

	def predict_one(self, record: Mapping[str, Any]) -> np.ndarray:
		fire_name = str(record["fire_name"])
		input_indices = tuple(int(value) for value in record["input_indices"])
		current_index = int(record["current_index"])
		# Validate the anti-leakage boundary on every record before using the cache.
		if not input_indices or input_indices[-1] != current_index:
			raise ValueError("Deterministic baseline record does not end at current_index.")
		if "target_index" in record and int(record["target_index"]) <= current_index:
			raise ValueError("Target index is not in the future.")
		prediction = self._full_prediction(fire_name, input_indices, current_index)
		return _crop_chw(prediction, record["patch"])

	def __call__(self, metadata_items: Sequence[Mapping[str, Any]]) -> torch.Tensor:
		predictions = [self.predict_one(record) for record in metadata_items]
		return torch.from_numpy(np.stack(predictions).astype(np.float32, copy=False))


class ProcessedTargetOnlyDataset(Dataset):
	"""Load canonical targets/metadata without reading model input or future raw frames."""

	def __init__(self, dataset_root: str | Path, sample_index_path: str | Path, split: str) -> None:
		self.root = Path(dataset_root).expanduser().resolve()
		self.sample_index_path = Path(sample_index_path).expanduser().resolve()
		self.split = str(split)
		self.records = [
			json.loads(line)
			for line in self.sample_index_path.read_text(encoding="utf-8").splitlines()
			if line.strip()
		]
		self.records = [record for record in self.records if str(record.get("split")) == self.split]
		if not self.records:
			raise ValueError(f"No processed records found for split={self.split!r}.")

	def __len__(self) -> int:
		return len(self.records)

	def __getitem__(self, index: int):
		record = self.records[index]
		patch = record["patch"]
		y0, x0, height, width = (int(patch[key]) for key in ("y0", "x0", "height", "width"))
		target_path = self.root / str(record["target_path"])
		with np.load(target_path, allow_pickle=False) as archive:
			target = np.stack(
				[
					np.asarray(archive[key], dtype=np.float32)[y0 : y0 + height, x0 : x0 + width]
					for key in ("surface_consumed", "canopy_consumed", "fire_mask", "energy_log")
				],
				axis=0,
			).astype(np.float32, copy=False)
		target[2] = (target[2] > 0.5).astype(np.float32)
		# The empty input tensor makes accidental model-style inference fail. The
		# deterministic evaluator passes only metadata to its predictor.
		return torch.empty(0, dtype=torch.float32), torch.from_numpy(target), dict(record)


__all__ = [
	"ProcessedHistoryBaselinePredictor",
	"ProcessedTargetOnlyDataset",
	"linear_extrapolation_from_observed_history",
	"load_observed_raw_history",
	"persistence_from_observed_history",
]
