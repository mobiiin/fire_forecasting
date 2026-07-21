from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.trim_prefire_frames import (
	detect_prefire_trim,
	frame_paths_for_record,
	trim_index,
	write_summary_csv,
)
from src.data.discovery import discover_multiple_datasets


def _write_frames(
	fire_dir: Path,
	num_frames: int,
	flux_active_idx: int | None = None,
	fuel_drop_idx: int | None = None,
) -> list[Path]:
	fire_dir.mkdir(parents=True, exist_ok=True)
	paths: list[Path] = []
	for index in range(num_frames):
		frame = np.zeros((4, 4, 86), dtype=np.float32)
		frame[..., 84] = 1.0
		frame[..., 85] = 1.0
		if flux_active_idx is not None and index >= flux_active_idx:
			frame[0, 0, 80] = 2.0
		if fuel_drop_idx is not None and index >= fuel_drop_idx:
			frame[1, 1, 84] = 0.99
		path = fire_dir / f"frame_{index:04d}.npy"
		np.save(path, frame)
		paths.append(path)
	return paths


def _index_for_fire(tmp_path: Path, fire_name: str = "TEST_FIRE", num_frames: int = 15, **frame_kwargs) -> Path:
	fire_dir = tmp_path / fire_name
	_write_frames(fire_dir, num_frames=num_frames, **frame_kwargs)
	index = {
		"main_data_dir": str(tmp_path),
		"num_fires": 1,
		"fires": {
			fire_name: {
				"fire_name": fire_name,
				"data_dir": str(fire_dir),
				"fire_root_dir": str(fire_dir),
				"file_pattern": "*.npy",
				"num_npy_files": num_frames,
				"valid_for_energy_release": True,
				"custom_metadata": "preserve_me",
			}
		},
	}
	path = tmp_path / "fire_dataset_index.json"
	path.write_text(json.dumps(index), encoding="utf-8")
	return path


def test_detects_flux_first_active_and_keeps_context(tmp_path: Path) -> None:
	paths = _write_frames(tmp_path / "fire", num_frames=15, flux_active_idx=10)

	result = detect_prefire_trim(paths, prefire_context_frames=3, flux_threshold=1.0, min_active_pixels=1)

	assert result["first_active_idx"] == 10
	assert result["trim_start_idx"] == 7
	assert result["first_active_reason"] == "flux"


def test_detects_fuel_consumption_first_active(tmp_path: Path) -> None:
	paths = _write_frames(tmp_path / "fire", num_frames=16, fuel_drop_idx=12)

	result = detect_prefire_trim(paths, prefire_context_frames=3, consumed_threshold=0.001, min_active_pixels=1)

	assert result["first_active_idx"] == 12
	assert result["trim_start_idx"] == 9
	assert result["first_active_reason"] == "consumed"


def test_no_activity_keeps_all_frames_and_warns(tmp_path: Path) -> None:
	paths = _write_frames(tmp_path / "fire", num_frames=12)

	result = detect_prefire_trim(paths, prefire_context_frames=3, min_active_pixels=1)

	assert result["first_active_idx"] is None
	assert result["trim_start_idx"] == 0
	assert result["trimmed_num_frames"] == 12
	assert result["warning"] == "no active frame detected"


def test_output_index_preserves_metadata_and_adds_prefire_trim(tmp_path: Path) -> None:
	input_index = _index_for_fire(tmp_path, num_frames=15, flux_active_idx=10)
	args = argparse.Namespace(
		input_index=str(input_index),
		main_data_dir=None,
		output_index=str(tmp_path / "trimmed.json"),
		prefire_context_frames=3,
		flux_threshold=1.0,
		consumed_threshold=0.001,
		min_active_pixels=1,
		mode="index_only",
		output_data_dir=None,
		dry_run=False,
		plot_diagnostics=False,
		diagnostics_dir=str(tmp_path / "diagnostics"),
		overwrite=False,
		file_pattern="*.npy",
		fire_dir_glob="*",
		recursive=True,
		input_sequence_length=6,
		prediction_horizon=1,
	)

	trimmed_index, rows, _diagnostics = trim_index(args)
	record = trimmed_index["fires"]["TEST_FIRE"]

	assert rows[0]["trim_start_idx"] == 7
	assert record["custom_metadata"] == "preserve_me"
	assert record["prefire_trim"]["first_active_idx"] == 10
	assert record["prefire_trim"]["trim_start_idx"] == 7
	assert record["num_npy_files"] == 8
	assert len(record["trimmed_frame_paths"]) == 8
	assert frame_paths_for_record(record)[0].name == "frame_0007.npy"


def test_trimmed_frame_paths_are_used_by_dataset_discovery(tmp_path: Path) -> None:
	input_index = _index_for_fire(tmp_path, num_frames=15, flux_active_idx=10)
	args = argparse.Namespace(
		input_index=str(input_index),
		main_data_dir=None,
		output_index=str(tmp_path / "trimmed.json"),
		prefire_context_frames=3,
		flux_threshold=1.0,
		consumed_threshold=0.001,
		min_active_pixels=1,
		mode="index_only",
		output_data_dir=None,
		dry_run=False,
		plot_diagnostics=False,
		diagnostics_dir=str(tmp_path / "diagnostics"),
		overwrite=False,
		file_pattern="*.npy",
		fire_dir_glob="*",
		recursive=True,
		input_sequence_length=6,
		prediction_horizon=1,
	)
	trimmed_index, _rows, _diagnostics = trim_index(args)
	trimmed_path = tmp_path / "fire_dataset_index_trimmed.json"
	trimmed_path.write_text(json.dumps(trimmed_index), encoding="utf-8")

	records = discover_multiple_datasets(
		{
			"main_data_dir": str(tmp_path),
			"fire_dataset_index_json": str(trimmed_path),
			"data_dirs": [],
			"data_discovery": {"mode": "fire_index"},
			"file_pattern": "*.npy",
			"use_patches": False,
			"patch_size": 64,
			"energy_release": {"enabled": False},
		}
	)

	assert len(records) == 1
	assert records[0]["file_paths"][0].name == "frame_0007.npy"
	assert len(records[0]["file_paths"]) == 8


def test_summary_csv_is_written(tmp_path: Path) -> None:
	rows = [
		{
			"fire_name": "TEST_FIRE",
			"original_num_frames": 15,
			"first_active_idx": 10,
			"trim_start_idx": 7,
			"removed_num_frames": 7,
			"trimmed_num_frames": 8,
			"first_active_reason": "flux",
			"max_flux_at_first_active": 2.0,
			"active_flux_pixels_at_first_active": 1,
			"active_consumed_pixels_at_first_active": 0,
			"warning": "",
		}
	]
	output_path = tmp_path / "prefire_trim_summary.csv"

	write_summary_csv(output_path, rows)

	with output_path.open(newline="", encoding="utf-8") as handle:
		loaded = list(csv.DictReader(handle))
	assert loaded[0]["fire_name"] == "TEST_FIRE"
	assert loaded[0]["trim_start_idx"] == "7"
