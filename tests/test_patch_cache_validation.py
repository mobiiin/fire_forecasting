from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.cache import MANIFEST_FILENAME, compute_cache_config_hash, validate_patch_cache


def _config(cache_dir: Path) -> dict:
	return {
		"config_path": str((cache_dir / "config.yaml").resolve()),
		"input_sequence_length": 2,
		"prediction_horizon": 1,
		"input_channel_count": 3,
		"patching": {
			"enabled": True,
			"patch_size": 4,
			"patch_height": 4,
			"patch_width": 4,
			"train_patch_mode": "sliding_window",
			"val_patch_mode": "sliding_window",
			"test_patch_mode": "sliding_window",
			"train_stride": 60,
			"val_stride": 60,
			"test_stride": 60,
			"include_border_patches": True,
		},
		"cache": {
			"enabled": True,
			"cache_dir": str(cache_dir),
			"cache_version": "v2_sliding_stride60",
			"use_precomputed_patches": True,
			"train_patch_mode": "sliding_window",
			"val_patch_mode": "sliding_window",
			"test_patch_mode": "sliding_window",
			"train_stride": 60,
			"val_stride": 60,
			"test_stride": 60,
		},
		"model": {
			"input_channels": 3,
			"output_channels": 2,
		},
		"task_type": "multitask",
	}


def _write_minimal_cache(cache_dir: Path, config: dict, *, train_stride: int = 60, train_patch_mode: str = "sliding_window") -> None:
	cache_dir.mkdir(parents=True, exist_ok=True)
	for split in ("train", "val", "test"):
		split_dir = cache_dir / split
		split_dir.mkdir(parents=True, exist_ok=True)
		np.savez(
			split_dir / "shard_000000.npz",
			X=np.zeros((1, 2, 3, 4, 4), dtype=np.float32),
			y=np.zeros((1, 2, 4, 4), dtype=np.float32),
		)
	manifest = {
		"cache_version": "v2_sliding_stride60",
		"created_at": "2026-07-10T00:00:00+00:00",
		"config_hash": compute_cache_config_hash(config),
		"input_sequence_length": 2,
		"input_channels": 3,
		"output_channels": 2,
		"patch_height": 4,
		"patch_width": 4,
		"include_border_patches": True,
		"patch_modes": {
			"train": train_patch_mode,
			"val": "sliding_window",
			"test": "sliding_window",
		},
		"strides": {
			"train": train_stride,
			"val": 60,
			"test": 60,
		},
		"num_train_patches": 1,
		"num_val_patches": 1,
		"num_test_patches": 1,
		"shards": {
			"train": [{"path": "train/shard_000000.npz", "num_samples": 1}],
			"val": [{"path": "val/shard_000000.npz", "num_samples": 1}],
			"test": [{"path": "test/shard_000000.npz", "num_samples": 1}],
		},
	}
	(cache_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_manifest_records_patch_modes_and_strides(tmp_path: Path) -> None:
	cache_dir = tmp_path / "cache"
	config = _config(cache_dir)
	_write_minimal_cache(cache_dir, config)
	summary = validate_patch_cache(config, split=["train", "val", "test"])
	assert summary["manifest"]["patch_modes"] == {
		"train": "sliding_window",
		"val": "sliding_window",
		"test": "sliding_window",
	}
	assert summary["manifest"]["strides"] == {
		"train": 60,
		"val": 60,
		"test": 60,
	}


def test_cache_validation_fails_on_stride_mismatch(tmp_path: Path) -> None:
	cache_dir = tmp_path / "cache"
	config = _config(cache_dir)
	_write_minimal_cache(cache_dir, config, train_stride=32)
	with pytest.raises(RuntimeError, match="different patch settings"):
		validate_patch_cache(config, split="train")


def test_cache_validation_fails_on_train_patch_mode_mismatch(tmp_path: Path) -> None:
	cache_dir = tmp_path / "cache"
	config = _config(cache_dir)
	_write_minimal_cache(cache_dir, config, train_patch_mode="single_sampled")
	with pytest.raises(RuntimeError, match="different patch settings"):
		validate_patch_cache(config, split="train")
