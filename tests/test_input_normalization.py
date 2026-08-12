from __future__ import annotations

import json
import re
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from scripts.compute_normalization import _save_stats
from src.data.preprocessing import load_normalization_stats
from src.training.input_normalization import (
	build_input_normalizer_for_loader,
	compare_normalization_metadata,
	normalization_metadata_from_loader,
	validate_normalization_stats,
)


class _Loader:
	def __init__(self, dataset) -> None:
		self.dataset = dataset


def test_device_input_normalizer_matches_zscore_math() -> None:
	torch = pytest.importorskip("torch")
	config = {"normalization": {"enabled": True, "input_normalization_device": "cuda"}}
	dataset = SimpleNamespace(
		input_normalization_on_device=True,
		inputs_are_normalized=False,
		normalization_stats={
			"mean": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
			"std": np.asarray([1.0, 2.0, 4.0], dtype=np.float32),
			"fit_split": np.asarray("train"),
		},
		config=config,
	)
	loader = _Loader(dataset)
	x = torch.arange(2 * 2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 2, 3, 2, 2)
	normalizer = build_input_normalizer_for_loader(loader, torch.device("cpu"), input_channels=3, config=config)

	actual = (x.clone() - normalizer["mean"]) / normalizer["std"]
	from src.training.input_normalization import apply_input_normalization

	torch.testing.assert_close(apply_input_normalization(x.clone(), normalizer), actual)


def test_normalization_stats_reject_channel_mismatch() -> None:
	stats = {
		"mean": np.asarray([0.0, 1.0], dtype=np.float32),
		"std": np.asarray([1.0, 1.0], dtype=np.float32),
		"fit_split": np.asarray("train"),
	}

	with pytest.raises(ValueError, match="channel count"):
		validate_normalization_stats(stats, input_channels=3, config={"normalization": {"enabled": True}})


def test_normalization_metadata_reports_loader_application_path() -> None:
	config = {"normalization": {"enabled": True, "input_normalization_device": "cuda"}}
	dataset = SimpleNamespace(
		input_normalization_on_device=True,
		inputs_are_normalized=False,
		normalization_stats={
			"mean": np.asarray([0.0, 1.0], dtype=np.float32),
			"std": np.asarray([1.0, 1.0], dtype=np.float32),
		},
	)
	metadata = normalization_metadata_from_loader(_Loader(dataset), config, input_channels=2)

	assert metadata["applied_by"] == "device"
	assert metadata["channel_count_matches"] is True


def test_checkpoint_normalization_metadata_mismatch_is_detected() -> None:
	mismatches = compare_normalization_metadata(
		{"enabled": True, "configured_device": "cpu", "applied_by": "dataset", "input_channels": 2},
		{"enabled": True, "configured_device": "device", "applied_by": "device", "input_channels": 2},
	)

	assert any("configured_device" in item for item in mismatches)
	assert any("applied_by" in item for item in mismatches)


def test_timestamped_normalization_save_creates_json_npz_and_latest_aliases(tmp_path: Path) -> None:
	config_path = tmp_path / "convlstm_soft_gate.yaml"
	config_path.write_text("experiment:\n  name: convlstm_soft_gate\n", encoding="utf-8")
	output_dir = tmp_path / "normalization"
	config = {
		"config_path": str(config_path),
		"experiment": {"name": "convlstm_soft_gate"},
		"input_sequence_length": 5,
		"prediction_horizon": 10,
		"cache": {"cache_version": "cache_v1", "cache_dir": str(tmp_path / "cache")},
	}
	stats = {
		"mean": np.asarray([1.0, 2.0], dtype=np.float32),
		"std": np.asarray([3.0, 4.0], dtype=np.float32),
		"min": np.asarray([0.0, 1.0], dtype=np.float32),
		"max": np.asarray([5.0, 6.0], dtype=np.float32),
	}

	paths = _save_stats(
		config_path,
		config,
		{"output_dir": str(output_dir), "apply_to_splits": ["train", "val", "test"]},
		stats,
		{"sample_count": 7, "pixel_count": 14, "input_channel_count": 2},
		latest_as_copy=True,
	)

	assert paths["json_path"].exists()
	assert paths["npz_path"].exists()
	assert paths["latest_json_path"].exists()
	assert paths["latest_npz_path"].exists()
	assert re.match(r"train_normalization_stats_convlstm_soft_gate_\d{8}_\d{6}\.json", paths["json_path"].name)
	assert re.match(r"train_normalization_stats_convlstm_soft_gate_\d{8}_\d{6}\.npz", paths["npz_path"].name)
	payload = json.loads(paths["json_path"].read_text(encoding="utf-8"))
	assert payload["normalization_version"] == "v2_timestamped_config_aware"
	assert payload["config"]["config_name"] == "convlstm_soft_gate"
	assert payload["config"]["config_path_absolute"] == str(config_path.resolve())
	assert payload["paths"]["npz_path"] == str(paths["npz_path"])
	assert payload["data"]["fit_split"] == "train"
	assert payload["cache"]["cache_version"] == "cache_v1"


def test_load_normalization_stats_follows_v2_json_to_npz(tmp_path: Path) -> None:
	npz_path = tmp_path / "train_normalization_stats_demo_20260805_170100.npz"
	json_path = tmp_path / "train_normalization_stats_demo_20260805_170100.json"
	np.savez_compressed(
		npz_path,
		mean=np.asarray([1.0, 2.0], dtype=np.float32),
		std=np.asarray([3.0, 4.0], dtype=np.float32),
		min=np.asarray([0.0, 1.0], dtype=np.float32),
		max=np.asarray([5.0, 6.0], dtype=np.float32),
	)
	json_path.write_text(
		json.dumps(
			{
				"normalization_version": "v2_timestamped_config_aware",
				"config": {"config_name": "demo", "config_path": "configs/experiments/demo.yaml"},
				"paths": {"npz_path": str(npz_path), "dataset_index_hash": "abc123"},
				"data": {"fit_split": "train", "input_channels": 2},
				"cache": {"cache_version": "cache_v1"},
			}
		),
		encoding="utf-8",
	)

	stats = load_normalization_stats(json_path)

	np.testing.assert_allclose(stats["mean"], np.asarray([1.0, 2.0], dtype=np.float32))
	assert str(stats["config_name"]) == "demo"
	assert str(stats["cache_version"]) == "cache_v1"
	assert str(stats["dataset_index_hash"]) == "abc123"


def test_load_normalization_stats_keeps_old_json_compatibility(tmp_path: Path) -> None:
	json_path = tmp_path / "old_stats.json"
	json_path.write_text(
		json.dumps(
			{
				"mean": [1.0, 2.0],
				"std": [3.0, 4.0],
				"min": [0.0, 1.0],
				"max": [5.0, 6.0],
				"fit_split": "train",
			}
		),
		encoding="utf-8",
	)

	stats = load_normalization_stats(json_path)

	np.testing.assert_allclose(stats["std"], np.asarray([3.0, 4.0]))


def test_load_normalization_stats_skips_object_metadata_in_npz(tmp_path: Path) -> None:
	npz_path = tmp_path / "normalization_stats.npz"
	np.savez_compressed(
		npz_path,
		mean=np.asarray([1.0, 2.0], dtype=np.float32),
		std=np.asarray([3.0, 4.0], dtype=np.float32),
		min=np.asarray([0.0, 1.0], dtype=np.float32),
		max=np.asarray([5.0, 6.0], dtype=np.float32),
		base_config_path=np.asarray(None, dtype=object),
	)

	stats = load_normalization_stats(npz_path)

	np.testing.assert_allclose(stats["mean"], np.asarray([1.0, 2.0], dtype=np.float32))
	assert "base_config_path" not in stats
