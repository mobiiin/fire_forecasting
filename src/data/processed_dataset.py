"""Small helpers for the staged, full-frame processed dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SPLITS = ("train", "val", "test")


def load_dataset_manifest(dataset_root: str | Path) -> dict[str, Any]:
	return json.loads((Path(dataset_root) / "dataset_manifest.json").read_text(encoding="utf-8"))


def load_channel_manifest(dataset_root: str | Path) -> dict[str, Any]:
	return json.loads((Path(dataset_root) / "channel_manifest.json").read_text(encoding="utf-8"))


def load_fire_manifest(dataset_root: str | Path, fire_name: str) -> dict[str, Any]:
	path = Path(dataset_root) / "fires" / fire_name / "fire_manifest.json"
	return json.loads(path.read_text(encoding="utf-8"))


def load_frame(dataset_root: str | Path, fire_name: str, local_index: int, array_key: str = "x_engineered") -> np.ndarray:
	path = Path(dataset_root) / "fires" / fire_name / "frames" / f"frame_{int(local_index):06d}.npz"
	with np.load(path, allow_pickle=False) as archive:
		if array_key not in archive:
			raise KeyError(f"{array_key!r} is not present in {path}; available={archive.files}")
		return np.asarray(archive[array_key])


def crop_patch(frame: np.ndarray, patch_record: Mapping[str, Any]) -> np.ndarray:
	array = np.asarray(frame)
	if array.ndim != 3:
		raise ValueError(f"crop_patch expects C,H,W, got {array.shape}")
	y0, x0 = int(patch_record["y0"]), int(patch_record["x0"])
	h, w = int(patch_record["height"]), int(patch_record["width"])
	patch = array[:, y0:y0 + h, x0:x0 + w]
	if patch.shape[1:] != (h, w):
		raise ValueError(f"Patch is out of bounds: frame={array.shape}, record={dict(patch_record)}")
	return patch


def load_split_fires(dataset_root: str | Path, split: str) -> list[str]:
	if split not in SPLITS:
		raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
	path = Path(dataset_root) / "indices" / "splits" / f"{split}_fires.json"
	return [str(name) for name in json.loads(path.read_text(encoding="utf-8"))]


def manifest_split_fires(manifest: Mapping[str, Any], split: str) -> list[str]:
	"""Read either canonical ``train/val/test`` or legacy ``*_fires`` keys."""
	splits = manifest.get("splits", {}) if isinstance(manifest, Mapping) else {}
	if not isinstance(splits, Mapping):
		return []
	value = splits.get(split, splits.get(f"{split}_fires", []))
	return [str(name) for name in value] if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def patch_starts(length: int, patch_size: int, stride: int, include_border: bool = True) -> list[int]:
	length, patch_size, stride = int(length), int(patch_size), int(stride)
	if length <= 0 or patch_size <= 0 or stride <= 0:
		raise ValueError("length, patch_size, and stride must be positive")
	if patch_size > length:
		raise ValueError(f"patch_size={patch_size} exceeds dimension length={length}")
	starts = list(range(0, length - patch_size + 1, stride))
	last = length - patch_size
	if include_border and starts[-1] != last:
		starts.append(last)
	return starts


def make_patch_id(fire_name: str, y0: int, x0: int, height: int, width: int) -> str:
	return f"{fire_name}_y{int(y0):03d}_x{int(x0):03d}_h{int(height):03d}_w{int(width):03d}"


def validate_split_assignments(
	index_fires: Mapping[str, Any],
	train_fires: Sequence[str],
	val_fires: Sequence[str],
	test_fires: Sequence[str],
) -> dict[str, list[str]]:
	assignments = {"train": [str(x) for x in train_fires], "val": [str(x) for x in val_fires], "test": [str(x) for x in test_fires]}
	seen: dict[str, str] = {}
	for split, names in assignments.items():
		if len(names) != len(set(names)):
			raise ValueError(f"Duplicate fire name in {split} split")
		for name in names:
			if name not in index_fires:
				raise ValueError(f"Fire {name!r} in {split} split is missing from fire_dataset_index")
			if name in seen:
				raise ValueError(f"Fire {name!r} overlaps {seen[name]} and {split} splits")
			seen[name] = split
	return assignments


def frame_npz_roundtrip(path: str | Path, x_engineered: np.ndarray, x_raw: np.ndarray | None = None, **metadata: Any) -> None:
	payload: dict[str, Any] = {"x_engineered": np.asarray(x_engineered, dtype=np.float32)}
	if x_raw is not None:
		payload["x_raw"] = np.asarray(x_raw, dtype=np.float32)
	for key, value in metadata.items():
		if isinstance(value, (int, np.integer)):
			payload[key] = np.asarray(int(value), dtype=np.int64)
		elif isinstance(value, (float, np.floating)):
			payload[key] = np.asarray(float(value), dtype=np.float32)
		else:
			payload[key] = np.asarray(str(value))
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	np.savez_compressed(path, **payload)
