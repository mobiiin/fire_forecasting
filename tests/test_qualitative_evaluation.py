from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.qualitative import (
	save_qualitative_summary_image,
	select_qualitative_sample_indices,
	selected_sample_record,
)


def test_select_qualitative_sample_indices_is_unique_and_reproducible() -> None:
	first = select_qualitative_sample_indices(dataset_length=100, num_samples=10, seed=42)
	second = select_qualitative_sample_indices(dataset_length=100, num_samples=10, seed=42)

	assert first == second
	assert len(first) == 10
	assert len(set(first)) == 10
	assert all(0 <= index < 100 for index in first)


def test_select_qualitative_sample_indices_caps_to_dataset_length() -> None:
	indices = select_qualitative_sample_indices(dataset_length=3, num_samples=10, seed=7)

	assert len(indices) == 3
	assert sorted(indices) == [0, 1, 2]


def test_selected_sample_record_keeps_sequence_and_patch_metadata() -> None:
	record = selected_sample_record(
		sample_number=2,
		dataset_index=17,
		metadata={
			"dataset_name": "WOOLSEY",
			"sample_index": 101,
			"input_indices": [101, 102, 103],
			"last_input_idx": 103,
			"target_idx": 113,
			"patch_top": 4,
			"patch_left": 8,
			"patch_size": 16,
		},
		split="test",
		input_sequence_length=3,
		prediction_horizon=10,
	)

	assert record["sample_number"] == 2
	assert record["dataset_index"] == 17
	assert record["fire_name"] == "WOOLSEY"
	assert record["local_sample_index"] == 101
	assert record["original_input_indices"] == [101, 102, 103]
	assert record["last_input_index"] == 103
	assert record["target_index"] == 113
	assert record["patch_coords"] == {"y0": 4, "y1": 20, "x0": 8, "x1": 24}
	assert record["input_sequence_length"] == 3
	assert record["prediction_horizon"] == 10


def test_qualitative_plot_handles_all_zero_single_model(tmp_path: Path) -> None:
	pytest.importorskip("matplotlib")
	target = np.zeros((4, 8, 8), dtype=np.float32)
	prediction = np.zeros((4, 8, 8), dtype=np.float32)
	output = tmp_path / "sample_000.png"

	save_qualitative_summary_image(
		target=target,
		predictions={"persistence": prediction},
		models=[{"key": "persistence", "display_name": "Persistence"}],
		sample_record={"sample_number": 0, "dataset_index": 0, "fire_name": "demo"},
		output_path=output,
		dpi=72,
	)

	assert output.exists()
	assert output.stat().st_size > 0


def test_qualitative_plot_writes_exactly_requested_summary_images(tmp_path: Path) -> None:
	pytest.importorskip("matplotlib")
	target = np.zeros((4, 8, 8), dtype=np.float32)
	prediction = np.zeros((4, 8, 8), dtype=np.float32)
	models = [
		{"key": "persistence", "display_name": "Persistence"},
		{"key": "linear_extrapolation", "display_name": "Linear Extrapolation"},
	]
	for sample_number in range(3):
		save_qualitative_summary_image(
			target=target,
			predictions={"persistence": prediction, "linear_extrapolation": prediction},
			models=models,
			sample_record={"sample_number": sample_number, "dataset_index": sample_number, "fire_name": "demo"},
			output_path=tmp_path / "images" / f"sample_{sample_number:03d}.png",
			dpi=72,
		)

	assert sorted(path.name for path in (tmp_path / "images").glob("sample_*.png")) == [
		"sample_000.png",
		"sample_001.png",
		"sample_002.png",
	]
