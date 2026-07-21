from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import manual_trim_fire_datasets as manual


def _write_frames(fire_dir: Path, count: int = 6) -> None:
	fire_dir.mkdir(parents=True, exist_ok=True)
	for index in range(count):
		frame = np.zeros((2, 2, 86), dtype=np.float32)
		frame[:, :, 84] = 1.0
		frame[:, :, 85] = 1.0
		path = fire_dir / f"frame_{index:04d}.npy"
		np.save(path, frame)


def _write_index(path: Path, count: int = 6) -> Path:
	fire_dir = path.parent / "FIRE_A"
	_write_frames(fire_dir, count=count)
	payload = {
		"num_fires": 1,
		"fires": {
			"FIRE_A": {
				"fire_name": "FIRE_A",
				"data_dir": str(fire_dir),
				"num_npy_files": count,
				"file_pattern": "*.npy",
			}
		},
	}
	path.write_text(json.dumps(payload), encoding="utf-8")
	return path


def _write_trim_config(path: Path, start: int = 2, end: int | None = None) -> Path:
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"version": "manual_fire_trim_v1",
		"source_index": "fire_dataset_index.json",
		"fires": {
			"FIRE_A": {
				"trim_start_index": start,
				"trim_end_index": end,
				"notes": "",
				"selected_with": "manual_trim_fire_datasets.py",
			}
		},
	}
	path.write_text(json.dumps(payload), encoding="utf-8")
	return path


def _args(tmp_path: Path, input_index: Path, trim_config: Path, output_index: Path | None = None) -> argparse.Namespace:
	return argparse.Namespace(
		input_index=str(input_index),
		output_index=str(output_index or tmp_path / "fire_dataset_index_trimmed.json"),
		trim_config=str(trim_config),
		start_fire=None,
		only_fire=None,
		skip_existing_choices=False,
		overwrite_existing_choices=False,
		no_overwrite=False,
		default_start_index=0,
		default_end_index=None,
		jump=10,
		save_every_choice=True,
		plot_diagnostics=False,
		diagnostics_dir=str(tmp_path / "diagnostics"),
		mode="terminal",
		apply_only=True,
		flux_threshold=1.0,
		consumed_threshold=0.001,
		min_active_pixels=5,
	)


def test_load_fire_index_handles_root_list_schema(tmp_path: Path) -> None:
	index_path = tmp_path / "index.json"
	index_path.write_text(json.dumps([{"name": "FIRE_A", "path": str(tmp_path / "FIRE_A")}]), encoding="utf-8")

	entries, root, schema = manual.load_fire_index(index_path)

	assert schema["kind"] == "root_list"
	assert isinstance(root, list)
	assert entries[0]["name"] == "FIRE_A"


def test_load_fire_index_handles_fires_list_schema(tmp_path: Path) -> None:
	index_path = tmp_path / "index.json"
	index_path.write_text(json.dumps({"fires": [{"fire_name": "FIRE_A", "path": str(tmp_path / "FIRE_A")}]}), encoding="utf-8")

	entries, _root, schema = manual.load_fire_index(index_path)

	assert schema["kind"] == "dict_list"
	assert schema["container_key"] == "fires"
	assert entries[0]["name"] == "FIRE_A"


def test_apply_only_writes_temporal_trim_without_adding_frame_paths(tmp_path: Path) -> None:
	input_index = _write_index(tmp_path / "fire_dataset_index.json", count=6)
	trim_config = _write_trim_config(tmp_path / "configs" / "manual_fire_trim.json", start=2, end=4)
	args = _args(tmp_path, input_index, trim_config)

	manual.run(args)
	output = json.loads(Path(args.output_index).read_text(encoding="utf-8"))
	record = output["fires"]["FIRE_A"]

	assert record["temporal_trim"]["trim_start_index"] == 2
	assert record["temporal_trim"]["trim_end_index"] == 4
	assert record["temporal_trim"]["trimmed_num_frames"] == 3
	assert "trimmed_frame_paths" not in record
	assert "frame_paths" not in record
	assert Path(tmp_path / "diagnostics" / "manual_trim_summary.csv").exists()
	assert Path(tmp_path / "diagnostics" / "manual_trim_summary.json").exists()


def test_output_index_overwrites_existing_by_default(tmp_path: Path) -> None:
	input_index = _write_index(tmp_path / "fire_dataset_index.json", count=6)
	trim_config = _write_trim_config(tmp_path / "configs" / "manual_fire_trim.json", start=1, end=3)
	output_index = tmp_path / "fire_dataset_index_trimmed.json"
	output_index.write_text("old content", encoding="utf-8")

	manual.run(_args(tmp_path, input_index, trim_config, output_index=output_index))

	output = json.loads(output_index.read_text(encoding="utf-8"))
	assert output["fires"]["FIRE_A"]["temporal_trim"]["trim_start_index"] == 1


def test_no_overwrite_prevents_replacing_existing_output(tmp_path: Path) -> None:
	input_index = _write_index(tmp_path / "fire_dataset_index.json", count=6)
	trim_config = _write_trim_config(tmp_path / "configs" / "manual_fire_trim.json", start=1, end=3)
	output_index = tmp_path / "fire_dataset_index_trimmed.json"
	output_index.write_text("old content", encoding="utf-8")
	args = _args(tmp_path, input_index, trim_config, output_index=output_index)
	args.no_overwrite = True

	with pytest.raises(FileExistsError):
		manual.run(args)


def test_null_trim_end_becomes_last_frame(tmp_path: Path) -> None:
	input_index = _write_index(tmp_path / "fire_dataset_index.json", count=6)
	trim_config = _write_trim_config(tmp_path / "configs" / "manual_fire_trim.json", start=2, end=None)
	args = _args(tmp_path, input_index, trim_config)

	manual.run(args)
	output = json.loads(Path(args.output_index).read_text(encoding="utf-8"))

	assert output["fires"]["FIRE_A"]["temporal_trim"]["trim_end_index"] == 5
	assert output["fires"]["FIRE_A"]["temporal_trim"]["trimmed_num_frames"] == 4


def test_invalid_trim_start_raises_clear_error(tmp_path: Path) -> None:
	input_index = _write_index(tmp_path / "fire_dataset_index.json", count=6)
	trim_config = _write_trim_config(tmp_path / "configs" / "manual_fire_trim.json", start=6, end=None)

	with pytest.raises(ValueError, match="trim_start_index"):
		manual.run(_args(tmp_path, input_index, trim_config))
