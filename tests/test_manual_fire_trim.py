from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.apply_manual_fire_trim import apply_manual_fire_trim
from scripts.precompute_patch_cache import _base_manifest
from src.data.cache import compute_current_trim_metadata_hash, compute_dataset_index_hash
from src.data.dataset import MultiFirePatchSequenceDataset
from src.data.fire_index import save_fire_dataset_index
from src.data.temporal_trim import max_valid_local_start


def _write_index(path: Path, num_frames: int = 120) -> Path:
	index = {
		"num_fires": 1,
		"fires": {
			"FIRE_A": {
				"fire_name": "FIRE_A",
				"data_dir": str(path.parent / "FIRE_A"),
				"num_npy_files": num_frames,
				"file_pattern": "*.npy",
				"valid_for_energy_release": True,
			}
		},
	}
	save_fire_dataset_index(index, path)
	return path


def _write_trim_config(path: Path, start: int = 10, end: int | None = 99) -> Path:
	payload = {
		"version": "manual_fire_trim_v1",
		"source_index": "fire_dataset_index.json",
		"fires": {
			"FIRE_A": {
				"trim_start_index": start,
				"trim_end_index": end,
				"notes": "",
				"selected_with": "visualize_input_dataset.py",
			}
		},
	}
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload), encoding="utf-8")
	return path


def _args(tmp_path: Path, input_index: Path, trim_config: Path) -> argparse.Namespace:
	return argparse.Namespace(
		input_index=str(input_index),
		trim_config=str(trim_config),
		output_index=str(tmp_path / "fire_dataset_index_trimmed.json"),
		default_start_index=0,
		default_end_index=None,
		input_sequence_length=5,
		prediction_horizon=10,
		require_all_fires=False,
		overwrite=False,
		dry_run=False,
		summary_csv=str(tmp_path / "summary.csv"),
	)


def test_apply_manual_fire_trim_writes_compact_temporal_metadata(tmp_path: Path) -> None:
	input_index = _write_index(tmp_path / "fire_dataset_index.json")
	trim_config = _write_trim_config(tmp_path / "configs" / "manual_fire_trim.json", start=10, end=99)

	trimmed_index, rows, summary = apply_manual_fire_trim(_args(tmp_path, input_index, trim_config))
	record = trimmed_index["fires"]["FIRE_A"]

	assert record["temporal_trim"]["trim_start_index"] == 10
	assert record["temporal_trim"]["trim_end_index"] == 99
	assert record["temporal_trim"]["original_num_frames"] == 120
	assert record["temporal_trim"]["trimmed_num_frames"] == 90
	assert "trimmed_frame_paths" not in record
	assert "original_frame_paths" not in record
	assert rows[0]["trimmed_num_frames"] == 90
	assert summary["fires_manually_trimmed"] == 1


def test_apply_manual_fire_trim_rejects_invalid_start(tmp_path: Path) -> None:
	input_index = _write_index(tmp_path / "fire_dataset_index.json", num_frames=20)
	trim_config = _write_trim_config(tmp_path / "configs" / "manual_fire_trim.json", start=20, end=None)

	with pytest.raises(ValueError, match="trim_start_index"):
		apply_manual_fire_trim(_args(tmp_path, input_index, trim_config))


def _write_frame_sequence(fire_dir: Path, num_frames: int) -> list[Path]:
	fire_dir.mkdir(parents=True, exist_ok=True)
	paths: list[Path] = []
	for index in range(num_frames):
		frame = np.zeros((2, 2, 86), dtype=np.float32)
		frame[:, :, 0] = float(index)
		path = fire_dir / f"frame_{index:04d}.npy"
		np.save(path, frame)
		paths.append(path)
	return paths


def test_dataset_maps_local_trimmed_sample_to_original_indices(tmp_path: Path) -> None:
	pytest.importorskip("torch")
	file_paths = _write_frame_sequence(tmp_path / "FIRE_A", num_frames=30)
	record = {
		"dataset_id": 0,
		"dataset_name": "FIRE_A",
		"data_dir": tmp_path / "FIRE_A",
		"file_paths": file_paths,
		"num_files": len(file_paths),
		"raw_shape": (2, 2, 86),
		"temporal_trim": {
			"enabled": True,
			"trim_start_index": 10,
			"trim_end_index": 24,
			"original_num_frames": 30,
			"trimmed_num_frames": 15,
		},
	}
	config = {
		"task_type": "regression",
		"target_channel": 0,
		"input_channel_count": 1,
		"input_sequence_length": 5,
		"prediction_horizon": 10,
		"model": {"output_channels": 1},
		"channel_layout": {
			"flux_channels": [80, 81, 82, 83],
			"fuel_channels": [84, 85],
			"surface_fuel_channel": 84,
			"canopy_fuel_channel": 85,
			"flux_mask_channel": 80,
		},
		"patching": {"enabled": False},
		"energy_release": {"enabled": False},
	}

	assert max_valid_local_start(record, 5, 10) == 0
	dataset = MultiFirePatchSequenceDataset(
		dataset_records=[record],
		sample_refs=[{"dataset_id": 0, "dataset_name": "FIRE_A", "sample_index": 0}],
		input_sequence_length=5,
		prediction_horizon=10,
		target_channel=0,
		input_channel_count=1,
		task_type="regression",
		return_metadata=True,
		config=config,
	)

	x_tensor, y_tensor, metadata = dataset[0]

	assert metadata["local_start_idx"] == 0
	assert metadata["original_start_idx"] == 10
	assert metadata["original_input_indices"] == [10, 11, 12, 13, 14]
	assert metadata["original_last_input_idx"] == 14
	assert metadata["original_target_idx"] == 24
	assert metadata["trim_start_index"] == 10
	assert metadata["trim_end_index"] == 24
	assert x_tensor[:, 0, 0, 0].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
	assert float(y_tensor[0, 0, 0]) == 24.0


def test_cache_manifest_records_dataset_and_trim_hashes(tmp_path: Path) -> None:
	index_path = _write_index(tmp_path / "fire_dataset_index_trimmed.json", num_frames=30)
	trimmed = json.loads(index_path.read_text(encoding="utf-8"))
	trimmed["fires"]["FIRE_A"]["temporal_trim"] = {
		"enabled": True,
		"trim_start_index": 10,
		"trim_end_index": 24,
		"original_num_frames": 30,
		"trimmed_num_frames": 15,
	}
	index_path.write_text(json.dumps(trimmed), encoding="utf-8")
	file_paths = _write_frame_sequence(tmp_path / "FIRE_A", num_frames=30)
	record = {
		"dataset_id": 0,
		"dataset_name": "FIRE_A",
		"data_dir": tmp_path / "FIRE_A",
		"file_paths": file_paths,
		"num_files": len(file_paths),
		"effective_num_files": 15,
		"raw_shape": (2, 2, 86),
		"temporal_trim": trimmed["fires"]["FIRE_A"]["temporal_trim"],
	}
	config = {
		"config_path": str(tmp_path / "config.yaml"),
		"fire_dataset_index_json": str(index_path),
		"split_mode": "manual_fire_holdout",
		"manual_fire_split": {},
		"input_sequence_length": 5,
		"prediction_horizon": 10,
		"target_channel": 0,
		"input_channel_count": 1,
		"task_type": "regression",
		"model": {"output_channels": 1},
		"channel_layout": {
			"flux_channels": [80, 81, 82, 83],
			"fuel_channels": [84, 85],
			"surface_fuel_channel": 84,
			"canopy_fuel_channel": 85,
			"flux_mask_channel": 80,
		},
		"patching": {"enabled": False},
		"cache": {"cache_version": "test"},
		"energy_release": {"enabled": False},
	}
	dataset = MultiFirePatchSequenceDataset(
		dataset_records=[record],
		sample_refs=[{"dataset_id": 0, "dataset_name": "FIRE_A", "sample_index": 0}],
		input_sequence_length=5,
		prediction_horizon=10,
		target_channel=0,
		input_channel_count=1,
		task_type="regression",
		return_metadata=True,
		config=config,
	)

	manifest = _base_manifest(config, [record], {"train": [], "val": [], "test": []}, dataset)

	assert manifest["dataset_index_hash"] == compute_dataset_index_hash(config)
	assert manifest["trim_metadata_hash"] == compute_current_trim_metadata_hash(config)
	assert manifest["temporal_trim_enabled"] is True
	assert manifest["fires"]["FIRE_A"]["trim_start_index"] == 10
